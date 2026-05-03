
"""
Tests for the trade reconstruction helpers.

These checks cover the current reconstruction rules for filtering actionable
tweets and pairing opposite signals into completed trades.

Run with: `python -m pytest tests/test_reconstruct.py`
"""

from datetime import datetime

from src.app.models import TweetRecord
from src.app.reconstruct import filter_trade_tweets, reconstruct_trades


def make_tweet(
    text: str,
    created_at: datetime,
    sentiment: str | None = None,
    symbol: str | None = None,
) -> TweetRecord:
    """Build a TweetRecord with concise syntax for test setup."""

    return TweetRecord(
        text=text,
        created_at=created_at,
        sentiment=sentiment,
        symbol=symbol,
    )


def test_filter_trade_tweets_excludes_neutral_and_missing_symbol():
    """Exclude neutral and symbol-less tweets, then sort the remainder by time."""

    tweets = [
        make_tweet(
            "Bullish without symbol",
            datetime(2024, 1, 1, 9, 35),
            sentiment="bullish",
            symbol=None,
        ),
        make_tweet(
            "Bearish AAPL",
            datetime(2024, 1, 1, 9, 40),
            sentiment="bearish",
            symbol="AAPL",
        ),
        make_tweet(
            "Neutral MSFT",
            datetime(2024, 1, 1, 9, 30),
            sentiment="neutral",
            symbol="MSFT",
        ),
        make_tweet(
            "Bullish TSLA",
            datetime(2024, 1, 1, 9, 20),
            sentiment="bullish",
            symbol="TSLA",
        ),
    ]

    filtered = filter_trade_tweets(tweets)

    # Only bullish/bearish tweets with symbols should remain.
    assert [tweet.text for tweet in filtered] == ["Bullish TSLA", "Bearish AAPL"]
    assert [tweet.created_at for tweet in filtered] == [
        datetime(2024, 1, 1, 9, 20),
        datetime(2024, 1, 1, 9, 40),
    ]


def test_reconstruct_trades_collapses_repeated_bullish_signals():
    """Treat repeated bullish tweets as one open long until sentiment flips."""

    first_bullish = make_tweet(
        "Bullish AAPL 1",
        datetime(2024, 1, 1, 9, 30),
        sentiment="bullish",
        symbol="AAPL",
    )
    second_bullish = make_tweet(
        "Bullish AAPL 2",
        datetime(2024, 1, 1, 9, 35),
        sentiment="bullish",
        symbol="AAPL",
    )
    third_bullish = make_tweet(
        "Bullish AAPL 3",
        datetime(2024, 1, 1, 9, 40),
        sentiment="bullish",
        symbol="AAPL",
    )
    bearish = make_tweet(
        "Bearish AAPL",
        datetime(2024, 1, 1, 10, 0),
        sentiment="bearish",
        symbol="AAPL",
    )

    trades = reconstruct_trades(
        [first_bullish, second_bullish, third_bullish, bearish]
    )

    # Later bullish tweets should be ignored while the long is already open.
    assert len(trades) == 1
    assert trades[0].direction == "long"
    assert trades[0].entry_tweet == first_bullish
    assert trades[0].exit_tweet == bearish


def test_reconstruct_trades_builds_short_trade():
    """Produce a short trade when a bearish signal is closed by a bullish one."""

    bearish = make_tweet(
        "Bearish TSLA",
        datetime(2024, 1, 2, 9, 30),
        sentiment="bearish",
        symbol="TSLA",
    )
    bullish = make_tweet(
        "Bullish TSLA",
        datetime(2024, 1, 2, 10, 0),
        sentiment="bullish",
        symbol="TSLA",
    )

    trades = reconstruct_trades([bearish, bullish])

    assert len(trades) == 1
    assert trades[0].direction == "short"


def test_reconstruct_trades_close_signal_opens_next_trade():
    """Use a closing signal as the next opening signal when sentiment flips."""

    bullish = make_tweet(
        "Bullish NVDA",
        datetime(2024, 1, 3, 9, 30),
        sentiment="bullish",
        symbol="NVDA",
    )
    bearish = make_tweet(
        "Bearish NVDA",
        datetime(2024, 1, 3, 10, 0),
        sentiment="bearish",
        symbol="NVDA",
    )
    repeated_bearish = make_tweet(
        "Bearish NVDA again",
        datetime(2024, 1, 3, 10, 30),
        sentiment="bearish",
        symbol="NVDA",
    )

    trades = reconstruct_trades([bullish, bearish, repeated_bearish])

    # The first bearish tweet closes the long trade and also opens a short.
    # The repeated bearish tweet does not close that short, so only one trade is completed.
    assert len(trades) == 1
    assert trades[0].exit_tweet == bearish
    assert all(trade.entry_tweet != repeated_bearish for trade in trades)
    assert all(trade.exit_tweet != repeated_bearish for trade in trades)


def test_reconstruct_trades_excludes_unfinished_last_trade():
    """Drop any final position that never receives an opposing closing signal."""

    tweets = [
        make_tweet(
            "Bullish AAPL",
            datetime(2024, 1, 4, 9, 30),
            sentiment="bullish",
            symbol="AAPL",
        )
    ]

    trades = reconstruct_trades(tweets)

    assert trades == []


def test_reconstruct_trades_two_completed_trades_on_double_flip():
    """Produce two completed trades when the same symbol flips twice."""

    first_bullish = make_tweet(
        "Bullish NVDA",
        datetime(2024, 1, 5, 9, 30),
        sentiment="bullish",
        symbol="NVDA",
    )
    bearish = make_tweet(
        "Bearish NVDA",
        datetime(2024, 1, 5, 10, 0),
        sentiment="bearish",
        symbol="NVDA",
    )
    final_bullish = make_tweet(
        "Bullish NVDA again",
        datetime(2024, 1, 5, 10, 30),
        sentiment="bullish",
        symbol="NVDA",
    )

    trades = reconstruct_trades([first_bullish, bearish, final_bullish])

    assert len(trades) == 2
    assert trades[0].direction == "long"
    assert trades[0].entry_tweet == first_bullish
    assert trades[0].exit_tweet == bearish
    assert trades[1].direction == "short"
    assert trades[1].entry_tweet == bearish
    assert trades[1].exit_tweet == final_bullish
