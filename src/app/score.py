
"""
Score reconstructed trades using daily price data.

This module takes completed trade signal pairs and attaches simple,
deterministic price-based performance metrics. For this MVP, only daily OHLCV
data is available, so entry uses the first available daily open on or after
the entry date and exit uses the first available daily close on or after the
exit date.

The goal is to keep scoring easy to read and easy to reason about:
- look up a daily open for the entry tweet
- look up a daily close for the exit tweet
- calculate return, hold duration, and correctness
- skip trades that cannot be scored because data is missing
"""

from datetime import datetime

import pandas as pd

from .models import TradeRecord
from .reconstruct import TradeSignalPair


def get_price_on_or_after(
    timestamp: datetime,
    price_df: pd.DataFrame,
    price_column: str = "close",
) -> float | None:
    """
    Returns a daily price for the first trading day on or after a timestamp.

    The lookup is based on the calendar date of the timestamp rather than the
    exact time of day. This matches the MVP rule that, because only daily OHLCV
    data is available, entry uses the first available daily open on or after
    the entry date and exit uses the first available daily close on or after
    the exit date.

    Returns `None` when the DataFrame has no row on or after the target date.
    """

    # Convert timestamp to a date so we match against daily OHLCV data
    target_date = pd.Timestamp(timestamp.date())

    # Filter to rows on or after that date (first valid trading day)
    matching_rows = price_df.loc[price_df["date"] >= target_date]

    if matching_rows.empty:
        return None

    return float(matching_rows.iloc[0][price_column])


def calculate_return_pct(direction: str, entry_price: float, exit_price: float) -> float:
    """
    Calculates the realized return percentage for a long or short trade.

    Long trades profit when price rises from entry to exit. Short trades profit
    when price falls from entry to exit.
    """

    # Short trades profit when price decreases
    # So the return calculation is inverted compared to longs
    if direction == "short":
        return ((entry_price - exit_price) / entry_price) * 100

    return ((exit_price - entry_price) / entry_price) * 100


def calculate_hold_duration_days(entry_time: datetime, exit_time: datetime) -> float:
    """
    Returns the elapsed time between entry and exit in days.

    The result is a float so partial days are preserved.
    """

    duration = exit_time - entry_time
    return duration.total_seconds() / 86400


def score_trade(
    trade_pair: TradeSignalPair,
    price_data: dict[str, pd.DataFrame],
) -> TradeRecord | None:
    """
    Scores one reconstructed trade using the available price history.

    Returns `None` when the symbol is missing from the price data or when
    either price lookup cannot be completed.

    A trade is considered correct only when its realized return percentage is
    strictly positive. Flat trades are counted as incorrect.
    """

    # Look up price history for the trade's symbol
    price_df = price_data.get(trade_pair.symbol)
    if price_df is None:
        return None

    entry_price = get_price_on_or_after(trade_pair.entry_time, price_df, price_column="open")
    exit_price = get_price_on_or_after(trade_pair.exit_time, price_df, price_column="close")

    # If either side of the trade cannot be mapped to a trading day, the
    # trade cannot be scored deterministically.
    if entry_price is None or exit_price is None:
        return None

    # Compute performance metrics for the completed trade
    return_pct = calculate_return_pct(
        trade_pair.direction,
        entry_price,
        exit_price,
    )
    hold_duration = calculate_hold_duration_days(
        trade_pair.entry_time,
        trade_pair.exit_time,
    )

    return TradeRecord(
        symbol=trade_pair.symbol,
        direction=trade_pair.direction,
        entry_tweet=trade_pair.entry_tweet,
        exit_tweet=trade_pair.exit_tweet,
        entry_time=trade_pair.entry_time,
        exit_time=trade_pair.exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        return_pct=return_pct,
        hold_duration=hold_duration,
        is_correct=return_pct > 0,
    )


def score_trades(
    trade_pairs: list[TradeSignalPair],
    price_data: dict[str, pd.DataFrame],
) -> list[TradeRecord]:
    """
    Scores a list of reconstructed trades and skips any unscorable entries.

    A trade is skipped when its symbol is missing from the price data or when
    an entry or exit price cannot be found.
    """

    scored_trades = []

    for trade_pair in trade_pairs:
        scored_trade = score_trade(trade_pair, price_data)
        # Only include trades that could be successfully scored with price data
        if scored_trade is not None:
            scored_trades.append(scored_trade)

    return scored_trades


def build_summary(trades: list[TradeRecord]) -> dict:
    """
    Builds a simple aggregate summary from scored trade records.

    When no trades are provided, counts are returned as zero and all rate or
    average metrics are returned as `0.0`.
    """

    completed_trade_count = len(trades)

    # Avoid division by zero and return a clean empty summary
    if completed_trade_count == 0:
        return {
            "completed_trade_count": 0,
            "correct_trade_count": 0,
            "incorrect_trade_count": 0,
            "completed_trade_accuracy": 0.0,
            "average_return_pct": 0.0,
            "average_hold_duration_days": 0.0,
        }

    correct_trade_count = sum(1 for trade in trades if trade.is_correct)
    incorrect_trade_count = completed_trade_count - correct_trade_count
    average_return_pct = sum(trade.return_pct for trade in trades) / completed_trade_count
    average_hold_duration_days = (
        sum(trade.hold_duration for trade in trades) / completed_trade_count
    )

    return {
        "completed_trade_count": completed_trade_count,
        "correct_trade_count": correct_trade_count,
        "incorrect_trade_count": incorrect_trade_count,
        "completed_trade_accuracy": (correct_trade_count / completed_trade_count) * 100,
        "average_return_pct": average_return_pct,
        "average_hold_duration_days": average_hold_duration_days,
    }
