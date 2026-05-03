
"""
Tests for the trade scoring helpers.

These checks cover the current MVP scoring rules for attaching daily entry and
exit prices to reconstructed trades and summarizing the completed results.

Run with: `python -m pytest tests/test_score.py`
"""

from datetime import datetime, timedelta

import pandas as pd

from src.app.models import TradeRecord, TweetRecord
from src.app.reconstruct import TradeSignalPair
from src.app.score import (
    build_summary,
    calculate_hold_duration_days,
    calculate_return_pct,
    get_price_on_or_after,
    score_trade,
    score_trades,
)


def make_tweet(
    text: str,
    created_at: datetime,
    sentiment: str | None = None,
    symbol: str | None = None,
) -> TweetRecord:
    """Builds a TweetRecord with concise syntax for test setup."""

    return TweetRecord(
        text=text,
        created_at=created_at,
        sentiment=sentiment,
        symbol=symbol,
    )


def make_trade_pair(
    symbol: str,
    direction: str,
    entry_time: datetime,
    exit_time: datetime,
    entry_sentiment: str,
    exit_sentiment: str,
) -> TradeSignalPair:
    """Builds a TradeSignalPair with matching entry and exit tweets."""

    entry_tweet = make_tweet(
        f"{entry_sentiment.title()} {symbol}",
        entry_time,
        sentiment=entry_sentiment,
        symbol=symbol,
    )
    exit_tweet = make_tweet(
        f"{exit_sentiment.title()} {symbol}",
        exit_time,
        sentiment=exit_sentiment,
        symbol=symbol,
    )

    return TradeSignalPair(
        symbol=symbol,
        direction=direction,
        entry_tweet=entry_tweet,
        exit_tweet=exit_tweet,
        entry_time=entry_time,
        exit_time=exit_time,
    )


def make_price_df(rows: list[tuple]) -> pd.DataFrame:
    """Builds a small daily OHLCV DataFrame for tests."""

    columns = ["date", "open", "high", "low", "close", "volume"]

    # Always create DataFrame with expected columns, even if rows is empty
    price_df = pd.DataFrame(rows, columns=columns)

    if not price_df.empty:
        price_df["date"] = pd.to_datetime(price_df["date"])
    else:
        # Ensure correct dtype for empty DataFrame
        price_df = price_df.astype(
            {
                "date": "datetime64[ns]",
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "float",
            }
        )

    return price_df


def test_get_price_on_or_after_returns_first_matching_close():
    """Returns the first available closing price on or after the target date."""

    price_df = make_price_df(
        [
            ("2024-01-02", 99.0, 102.0, 98.0, 100.0, 1_000),
            ("2024-01-03", 104.0, 106.0, 103.0, 105.0, 1_100),
            ("2024-01-04", 107.0, 109.0, 106.0, 108.0, 1_200),
        ]
    )

    price = get_price_on_or_after(datetime(2024, 1, 3, 8, 0), price_df)

    assert price == 105.0


def test_get_price_on_or_after_returns_requested_price_column():
    """Returns the requested daily price column for the first matching row."""

    price_df = make_price_df(
        [
            ("2024-01-02", 99.0, 102.0, 98.0, 100.0, 1_000),
            ("2024-01-03", 104.0, 106.0, 103.0, 105.0, 1_100),
        ]
    )

    price = get_price_on_or_after(
        datetime(2024, 1, 3, 8, 0),
        price_df,
        price_column="open",
    )

    assert price == 104.0


def test_get_price_on_or_after_returns_none_when_no_future_price_exists():
    """Returns None when no row exists on or after the requested date."""

    price_df = make_price_df(
        [
            ("2024-01-02", 99.0, 102.0, 98.0, 100.0, 1_000),
            ("2024-01-03", 104.0, 106.0, 103.0, 105.0, 1_100),
        ]
    )

    price = get_price_on_or_after(datetime(2024, 1, 5, 9, 30), price_df)

    assert price is None


def test_calculate_return_pct_for_long_trade():
    """Calculates the expected positive return for a long trade."""

    assert calculate_return_pct("long", 100.0, 110.0) == 10.0


def test_calculate_return_pct_for_short_trade():
    """Calculates the expected positive return for a short trade."""

    assert calculate_return_pct("short", 100.0, 90.0) == 10.0


def test_calculate_hold_duration_days_returns_fractional_days():
    """Preserves partial days when computing hold duration."""

    entry_time = datetime(2024, 1, 1, 9, 0)
    exit_time = entry_time + timedelta(days=1, hours=12)

    duration = calculate_hold_duration_days(entry_time, exit_time)

    assert duration == 1.5


def test_score_trade_builds_trade_record_for_long_trade():
    """Builds a scored long trade using daily open for entry and close for exit."""

    trade_pair = make_trade_pair(
        symbol="AAPL",
        direction="long",
        entry_time=datetime(2024, 1, 2, 9, 30),
        exit_time=datetime(2024, 1, 4, 10, 0),
        entry_sentiment="bullish",
        exit_sentiment="bearish",
    )
    price_data = {
        "AAPL": make_price_df(
            [
                ("2024-01-02", 99.0, 102.0, 98.0, 100.0, 1_000),
                ("2024-01-03", 101.0, 104.0, 100.0, 103.0, 1_100),
                ("2024-01-04", 109.0, 111.0, 108.0, 110.0, 1_200),
            ]
        )
    }

    result = score_trade(trade_pair, price_data)

    assert result is not None
    assert result.symbol == "AAPL"
    assert result.direction == "long"
    assert result.entry_price == 99.0
    assert result.exit_price == 110.0
    assert result.return_pct == ((110.0 - 99.0) / 99.0) * 100
    assert result.is_correct is True


def test_score_trade_treats_flat_trade_as_incorrect():
    """Counts a flat trade as incorrect when return is exactly zero."""

    trade_pair = make_trade_pair(
        symbol="MSFT",
        direction="long",
        entry_time=datetime(2024, 2, 1, 9, 30),
        exit_time=datetime(2024, 2, 2, 10, 0),
        entry_sentiment="bullish",
        exit_sentiment="bearish",
    )
    price_data = {
        "MSFT": make_price_df(
            [
                ("2024-02-01", 200.0, 202.0, 198.0, 200.0, 900),
                ("2024-02-02", 199.0, 201.0, 197.0, 200.0, 950),
            ]
        )
    }

    result = score_trade(trade_pair, price_data)

    assert result is not None
    assert result.return_pct == 0.0
    assert result.is_correct is False


def test_score_trade_returns_none_when_symbol_is_missing():
    """Returns None when no price history exists for the trade symbol."""

    trade_pair = make_trade_pair(
        symbol="TSLA",
        direction="short",
        entry_time=datetime(2024, 3, 1, 9, 30),
        exit_time=datetime(2024, 3, 4, 9, 30),
        entry_sentiment="bearish",
        exit_sentiment="bullish",
    )

    result = score_trade(trade_pair, {"AAPL": make_price_df([])})

    assert result is None


def test_score_trades_skips_unscorable_trades():
    """Includes only trades that can be fully scored with available prices."""

    valid_trade = make_trade_pair(
        symbol="AAPL",
        direction="long",
        entry_time=datetime(2024, 1, 2, 9, 30),
        exit_time=datetime(2024, 1, 3, 9, 30),
        entry_sentiment="bullish",
        exit_sentiment="bearish",
    )
    missing_trade = make_trade_pair(
        symbol="TSLA",
        direction="short",
        entry_time=datetime(2024, 1, 2, 9, 30),
        exit_time=datetime(2024, 1, 3, 9, 30),
        entry_sentiment="bearish",
        exit_sentiment="bullish",
    )
    price_data = {
        "AAPL": make_price_df(
            [
                ("2024-01-02", 99.0, 102.0, 98.0, 100.0, 1_000),
                ("2024-01-03", 104.0, 106.0, 103.0, 105.0, 1_100),
            ]
        )
    }

    results = score_trades([valid_trade, missing_trade], price_data)

    assert len(results) == 1
    assert results[0].symbol == "AAPL"


def test_build_summary_returns_expected_metrics():
    """Aggregates counts, accuracy, and averages across completed trades."""

    trades = [
        TradeRecord(
            symbol="AAPL",
            direction="long",
            entry_tweet=make_tweet("Bullish AAPL", datetime(2024, 1, 2, 9, 30)),
            exit_tweet=make_tweet("Bearish AAPL", datetime(2024, 1, 4, 10, 0)),
            entry_time=datetime(2024, 1, 2, 9, 30),
            exit_time=datetime(2024, 1, 4, 10, 0),
            entry_price=100.0,
            exit_price=110.0,
            return_pct=10.0,
            hold_duration=2.0,
            is_correct=True,
        ),
        TradeRecord(
            symbol="TSLA",
            direction="short",
            entry_tweet=make_tweet("Bearish TSLA", datetime(2024, 1, 5, 9, 30)),
            exit_tweet=make_tweet("Bullish TSLA", datetime(2024, 1, 6, 9, 30)),
            entry_time=datetime(2024, 1, 5, 9, 30),
            exit_time=datetime(2024, 1, 6, 9, 30),
            entry_price=100.0,
            exit_price=105.0,
            return_pct=-5.0,
            hold_duration=1.0,
            is_correct=False,
        ),
    ]

    summary = build_summary(trades)

    assert summary == {
        "completed_trade_count": 2,
        "correct_trade_count": 1,
        "incorrect_trade_count": 1,
        "completed_trade_accuracy": 50.0,
        "average_return_pct": 2.5,
        "average_hold_duration_days": 1.5,
    }


def test_build_summary_returns_zero_values_for_empty_input():
    """Returns a clean zeroed summary when there are no completed trades."""

    summary = build_summary([])

    assert summary == {
        "completed_trade_count": 0,
        "correct_trade_count": 0,
        "incorrect_trade_count": 0,
        "completed_trade_accuracy": 0.0,
        "average_return_pct": 0.0,
        "average_hold_duration_days": 0.0,
    }
