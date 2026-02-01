"""
Solana Memecoin Wallet Analyzer
Identifies the most profitable wallets from Solana memecoin token runners
and detects "smart money" overlap across multiple tokens.

Usage:
    pip install streamlit requests pandas python-dotenv
    streamlit run solana_wallet_analyzer.py

Optionally create a .env file with:
    BIRDEYE_API_KEY=your_key_here
    HELIUS_API_KEY=your_key_here
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# Config
# ============================================================================

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
HELIUS_BASE_URL = "https://api.helius.xyz"

BIRDEYE_RATE_LIMIT_PER_SECOND = 5
HELIUS_RATE_LIMIT_PER_SECOND = 10

DEFAULT_TRADERS_PER_PAGE = 10
DEFAULT_PAGES_PER_TOKEN = 5
MAX_PAGES_PER_TOKEN = 10

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0

DEFAULT_TIME_FRAME = "24h"
TIME_FRAME_OPTIONS = ["30m", "1h", "2h", "4h", "8h", "24h"]
DEFAULT_OVERLAP_THRESHOLD = 2

# ============================================================================
# Models
# ============================================================================


@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    decimals: int = 9
    logo_uri: str = ""


@dataclass
class TraderRecord:
    wallet: str
    token_address: str
    token_symbol: str
    volume_buy: float = 0.0
    volume_sell: float = 0.0
    trade_count_buy: int = 0
    trade_count_sell: int = 0
    is_bot: bool = False

    @property
    def estimated_pnl(self) -> float:
        return self.volume_sell - self.volume_buy

    @property
    def total_volume(self) -> float:
        return self.volume_buy + self.volume_sell

    @property
    def total_trades(self) -> int:
        return self.trade_count_buy + self.trade_count_sell


@dataclass
class WalletOverlap:
    wallet: str
    token_count: int = 0
    tokens: List[str] = field(default_factory=list)
    total_estimated_pnl: float = 0.0
    total_volume: float = 0.0
    records: List[TraderRecord] = field(default_factory=list)


# ============================================================================
# API Clients
# ============================================================================


class RateLimiter:
    def __init__(self, calls_per_second: int):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.monotonic()


class BirdeyeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = BIRDEYE_BASE_URL
        self.rate_limiter = RateLimiter(BIRDEYE_RATE_LIMIT_PER_SECOND)
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "x-chain": "solana",
        })

    def _request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(MAX_RETRIES):
            self.rate_limiter.wait()
            try:
                resp = self.session.request(method, url, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def search_token(self, keyword: str) -> dict:
        return self._request("GET", "/defi/v3/search", params={
            "keyword": keyword,
            "chain": "solana",
        })

    def get_top_traders(
        self, token_address: str, time_frame: str = "24h", offset: int = 0, limit: int = 10
    ) -> dict:
        return self._request("GET", "/defi/v2/tokens/top_traders", params={
            "address": token_address,
            "time_frame": time_frame,
            "sort_by": "volume",
            "sort_type": "desc",
            "offset": offset,
            "limit": limit,
        })

    def get_token_metadata(self, token_address: str) -> dict:
        return self._request("GET", "/defi/v3/token/meta-data/single", params={
            "address": token_address,
        })


class HeliusClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = HELIUS_BASE_URL
        self.rate_limiter = RateLimiter(HELIUS_RATE_LIMIT_PER_SECOND)
        self.session = requests.Session()

    def _request(self, method: str, path: str, params: Optional[dict] = None, json_body: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        if params is None:
            params = {}
        params["api-key"] = self.api_key
        last_exc = None
        for attempt in range(MAX_RETRIES):
            self.rate_limiter.wait()
            try:
                resp = self.session.request(method, url, params=params, json=json_body, timeout=15)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def get_asset(self, token_address: str) -> dict:
        return self._request("POST", "/v0/token-metadata", json_body={
            "mintAccounts": [token_address],
            "includeOffChain": True,
        })


# ============================================================================
# Data Processing
# ============================================================================


def resolve_token(birdeye: BirdeyeClient, keyword: str) -> Optional[TokenInfo]:
    clean = keyword.strip().lstrip("$").lower()
    if not clean:
        return None
    try:
        data = birdeye.search_token(clean)
    except Exception as exc:
        st.warning(f"Search failed for '{clean}': {exc}")
        return None

    raw_items = data.get("data", {}).get("items", [])

    tokens_list = []
    for group in raw_items:
        if isinstance(group, dict) and "result" in group:
            tokens_list.extend(group["result"])
        elif isinstance(group, dict) and "address" in group:
            tokens_list.append(group)

    for token in tokens_list:
        addr = token.get("address", "")
        symbol = token.get("symbol", "").strip("$").lower()
        network = token.get("network", token.get("chain", "solana"))
        if network != "solana":
            continue
        if symbol == clean or clean in symbol:
            return TokenInfo(
                address=addr,
                symbol=token.get("symbol", clean.upper()).strip("$"),
                name=token.get("name", ""),
                decimals=token.get("decimals", 9),
                logo_uri=token.get("logo_uri", token.get("logoURI", "")),
            )

    if tokens_list:
        token = tokens_list[0]
        return TokenInfo(
            address=token.get("address", ""),
            symbol=token.get("symbol", clean.upper()).strip("$"),
            name=token.get("name", ""),
            decimals=token.get("decimals", 9),
            logo_uri=token.get("logo_uri", token.get("logoURI", "")),
        )

    return None


def _looks_like_address(text: str) -> bool:
    clean = text.strip()
    return 32 <= len(clean) <= 44 and clean.isalnum()


def resolve_token_with_fallback(
    birdeye: BirdeyeClient, helius: Optional[HeliusClient], keyword: str
) -> Optional[TokenInfo]:
    clean = keyword.strip()

    if _looks_like_address(clean):
        try:
            data = birdeye.get_token_metadata(clean)
            meta = data.get("data", {})
            if meta and meta.get("address"):
                return TokenInfo(
                    address=meta.get("address", clean),
                    symbol=meta.get("symbol", "UNKNOWN"),
                    name=meta.get("name", ""),
                    decimals=meta.get("decimals", 9),
                    logo_uri=meta.get("logo", ""),
                )
        except Exception:
            pass
        if helius:
            try:
                result = helius.get_asset(clean)
                if isinstance(result, list) and result:
                    item = result[0]
                    onchain = item.get("onChainMetadata", {}).get("metadata", {}).get("data", {})
                    offchain = item.get("offChainMetadata", {}).get("metadata", {})
                    return TokenInfo(
                        address=clean,
                        symbol=onchain.get("symbol", offchain.get("symbol", "UNKNOWN")),
                        name=onchain.get("name", offchain.get("name", "")),
                    )
            except Exception:
                pass
        return None

    token = resolve_token(birdeye, keyword)
    if token and token.address:
        return token

    return None


def fetch_top_traders(
    birdeye: BirdeyeClient,
    token: TokenInfo,
    time_frame: str,
    pages: int,
    exclude_bots: bool = False,
) -> List[TraderRecord]:
    traders: List[TraderRecord] = []
    limit = DEFAULT_TRADERS_PER_PAGE

    for page in range(pages):
        offset = page * limit
        try:
            data = birdeye.get_top_traders(
                token_address=token.address,
                time_frame=time_frame,
                offset=offset,
                limit=limit,
            )
        except Exception as exc:
            st.warning(f"Failed to fetch traders page {page + 1} for {token.symbol}: {exc}")
            break

        items = data.get("data", {}).get("items", [])
        if not items:
            inner = data.get("data", {})
            if isinstance(inner, dict) and "traders" in inner:
                items = inner["traders"]
            if not items:
                break

        for item in items:
            is_bot = item.get("tags") is not None and "bot" in str(item.get("tags", "")).lower()
            if exclude_bots and is_bot:
                continue
            record = TraderRecord(
                wallet=item.get("owner", item.get("address", "")),
                token_address=token.address,
                token_symbol=token.symbol,
                volume_buy=float(item.get("volumeBuy", item.get("volume_buy", 0))),
                volume_sell=float(item.get("volumeSell", item.get("volume_sell", 0))),
                trade_count_buy=int(item.get("tradeBuy", item.get("trade_buy", 0))),
                trade_count_sell=int(item.get("tradeSell", item.get("trade_sell", 0))),
                is_bot=is_bot,
            )
            if record.wallet:
                traders.append(record)

    return traders


def detect_overlaps(
    all_traders: Dict[str, List[TraderRecord]], threshold: int = 2
) -> List[WalletOverlap]:
    wallet_map: Dict[str, List[TraderRecord]] = {}

    for token_symbol, records in all_traders.items():
        for rec in records:
            wallet_map.setdefault(rec.wallet, []).append(rec)

    overlaps: List[WalletOverlap] = []
    for wallet, records in wallet_map.items():
        unique_tokens = list({r.token_symbol for r in records})
        if len(unique_tokens) >= threshold:
            overlaps.append(WalletOverlap(
                wallet=wallet,
                token_count=len(unique_tokens),
                tokens=unique_tokens,
                total_estimated_pnl=sum(r.estimated_pnl for r in records),
                total_volume=sum(r.total_volume for r in records),
                records=records,
            ))

    overlaps.sort(key=lambda o: o.total_estimated_pnl, reverse=True)
    return overlaps


def traders_to_dataframe(traders: List[TraderRecord]) -> pd.DataFrame:
    if not traders:
        return pd.DataFrame()

    rows = []
    for t in traders:
        rows.append({
            "Wallet": t.wallet,
            "Buy Volume ($)": round(t.volume_buy, 2),
            "Sell Volume ($)": round(t.volume_sell, 2),
            "Est. PnL ($)": round(t.estimated_pnl, 2),
            "Total Volume ($)": round(t.total_volume, 2),
            "Buy Trades": t.trade_count_buy,
            "Sell Trades": t.trade_count_sell,
            "Bot": t.is_bot,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Est. PnL ($)", ascending=False).reset_index(drop=True)
    return df


def overlaps_to_dataframe(overlaps: List[WalletOverlap]) -> pd.DataFrame:
    if not overlaps:
        return pd.DataFrame()

    rows = []
    for o in overlaps:
        rows.append({
            "Wallet": o.wallet,
            "Tokens": ", ".join(o.tokens),
            "Token Count": o.token_count,
            "Total Est. PnL ($)": round(o.total_estimated_pnl, 2),
            "Total Volume ($)": round(o.total_volume, 2),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Solana Memecoin Wallet Analyzer", layout="wide")
st.title("Solana Memecoin Wallet Analyzer")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    birdeye_key = st.text_input(
        "Birdeye API Key",
        value=BIRDEYE_API_KEY,
        type="password",
        help="Required. Get one at birdeye.so",
    )
    helius_key = st.text_input(
        "Helius API Key (optional)",
        value=HELIUS_API_KEY,
        type="password",
        help="Optional fallback for token metadata",
    )

    st.divider()
    time_frame = st.selectbox(
        "Time Frame",
        TIME_FRAME_OPTIONS,
        index=TIME_FRAME_OPTIONS.index(DEFAULT_TIME_FRAME),
    )
    pages_per_token = st.slider(
        "Pages per Token",
        min_value=1,
        max_value=MAX_PAGES_PER_TOKEN,
        value=DEFAULT_PAGES_PER_TOKEN,
        help=f"Each page = 10 traders. Max {MAX_PAGES_PER_TOKEN} pages (100 traders).",
    )
    overlap_threshold = st.slider(
        "Overlap Threshold",
        min_value=2,
        max_value=10,
        value=DEFAULT_OVERLAP_THRESHOLD,
        help="Minimum number of tokens a wallet must appear in to count as overlap.",
    )
    exclude_bots = st.checkbox("Exclude Bots", value=False, help="Filter out wallets tagged as bots by Birdeye.")

# --- Main Area ---
st.subheader("Enter Token Tickers")
token_input = st.text_area(
    "One ticker per line (e.g. $elon, $crust) or paste contract addresses",
    height=120,
    placeholder="$elon\n$crust\n$wif",
)

analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

if analyze_clicked:
    if not birdeye_key:
        st.error("Please enter a Birdeye API key in the sidebar.")
        st.stop()

    raw_lines = [line.strip() for line in token_input.strip().splitlines() if line.strip()]
    if not raw_lines:
        st.error("Please enter at least one token ticker or address.")
        st.stop()

    birdeye = BirdeyeClient(birdeye_key)
    helius = HeliusClient(helius_key) if helius_key else None

    # Step 1: Resolve tokens
    resolved_tokens = {}
    progress = st.progress(0, text="Resolving tokens...")

    for i, keyword in enumerate(raw_lines):
        progress.progress((i + 1) / len(raw_lines), text=f"Resolving: {keyword}")
        token = resolve_token_with_fallback(birdeye, helius, keyword)
        if token and token.address:
            resolved_tokens[token.symbol] = token
            st.toast(f"Resolved {keyword} -> {token.symbol} ({token.address[:8]}...)")
        else:
            st.warning(f"Could not resolve '{keyword}' to a Solana token.")

    if not resolved_tokens:
        st.error("No tokens could be resolved. Check your input and API key.")
        st.stop()

    # Step 2: Fetch traders
    all_traders = {}
    total_steps = len(resolved_tokens)
    progress2 = st.progress(0, text="Fetching top traders...")

    for i, (symbol, token) in enumerate(resolved_tokens.items()):
        progress2.progress((i + 1) / total_steps, text=f"Fetching traders for {symbol}...")
        traders = fetch_top_traders(
            birdeye, token, time_frame, pages_per_token, exclude_bots
        )
        all_traders[symbol] = traders

    progress2.empty()

    # Step 3: Detect overlaps
    overlaps = detect_overlaps(all_traders, overlap_threshold)

    # Summary metrics
    unique_wallets = {r.wallet for records in all_traders.values() for r in records}
    col1, col2, col3 = st.columns(3)
    col1.metric("Tokens Analyzed", len(resolved_tokens))
    col2.metric("Unique Wallets", len(unique_wallets))
    col3.metric("Smart Money Overlaps", len(overlaps))

    # Tabbed results
    tab_names = list(resolved_tokens.keys()) + ["Smart Money Overlap"]
    tabs = st.tabs(tab_names)

    for idx, symbol in enumerate(resolved_tokens.keys()):
        with tabs[idx]:
            traders = all_traders.get(symbol, [])
            token = resolved_tokens[symbol]
            st.caption(f"{token.name} — `{token.address}`")

            if not traders:
                st.info(f"No trader data found for {symbol}.")
                continue

            df = traders_to_dataframe(traders)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Wallet": st.column_config.TextColumn("Wallet", width="large"),
                    "Est. PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Buy Volume ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Sell Volume ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Total Volume ($)": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

            profitable = sum(1 for t in traders if t.estimated_pnl > 0)
            st.caption(
                f"{len(traders)} traders loaded | "
                f"{profitable} profitable | "
                f"Top PnL: ${max(t.estimated_pnl for t in traders):,.2f}"
            )

    # Smart Money Overlap tab
    with tabs[-1]:
        if not overlaps:
            st.info(
                f"No wallets found trading {overlap_threshold}+ tokens. "
                "Try lowering the overlap threshold or adding more tokens."
            )
        else:
            overlap_df = overlaps_to_dataframe(overlaps)
            st.dataframe(
                overlap_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Wallet": st.column_config.TextColumn("Wallet", width="large"),
                    "Total Est. PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Total Volume ($)": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

            # Expandable detail per overlap wallet
            st.subheader("Wallet Details")
            for overlap in overlaps:
                short_wallet = f"{overlap.wallet[:6]}...{overlap.wallet[-4:]}"
                with st.expander(
                    f"{short_wallet} — {overlap.token_count} tokens — "
                    f"PnL: ${overlap.total_estimated_pnl:,.2f}"
                ):
                    for record in overlap.records:
                        st.markdown(
                            f"**{record.token_symbol}**: "
                            f"Buy ${record.volume_buy:,.2f} | "
                            f"Sell ${record.volume_sell:,.2f} | "
                            f"PnL ${record.estimated_pnl:,.2f} | "
                            f"Trades: {record.total_trades}"
                        )
