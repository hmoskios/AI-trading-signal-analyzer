
"""
Reconstruct completed trade signal pairs from classified tweet history.

This module turns a chronological stream of tweet-level trading signals into
completed trade "shells" that can be scored later with price data.

The goal here is to:
- keep only actionable tweet signals
- group signals by symbol
- pair an opening signal with the next opposite signal on that symbol

This module does not look up prices, calculate returns, or decide whether a
trade was profitable. It only determines the entry and exit signal pair for
each completed trade in a deterministic way.
"""

from dataclasses import dataclass
from datetime import datetime

from .models import TweetRecord


@dataclass
class TradeSignalPair:
    """
    Represents one completed trade reconstructed from two opposing tweet signals.

    A trade begins when a tweet opens a directional view on a symbol:
    - `bullish` opens a `long`
    - `bearish` opens a `short`

    The trade completes when a later tweet on the same symbol expresses the
    opposite sentiment. This dataclass is an intermediate structure used before
    later pipeline stages attach prices, returns, and correctness metrics.
    """

    symbol: str
    direction: str  # "long" or "short"

    entry_tweet: TweetRecord
    exit_tweet: TweetRecord

    entry_time: datetime
    exit_time: datetime


def filter_trade_tweets(tweets: list[TweetRecord]) -> list[TweetRecord]:
    """
    Returns only tweets that can participate in trade reconstruction.

    Trade reconstruction only uses tweets that express a directional view and
    identify a symbol to which that view applies.

    Neutral tweets are excluded because they do not open or close positions
    under the challenge rules. Tweets without a symbol are also excluded
    because reconstruction is performed independently per symbol, so a tweet
    with no symbol cannot be placed into a symbol-specific trade sequence.

    The returned list is sorted by `created_at` ascending so downstream
    reconstruction behaves deterministically regardless of input order.
    """

    filtered_tweets = []

    for tweet in tweets:
        if tweet.sentiment not in {"bullish", "bearish"}:
            continue

        if not tweet.symbol:
            continue

        filtered_tweets.append(tweet)

    return sorted(filtered_tweets, key=lambda tweet: tweet.created_at)


def reconstruct_trades(tweets: list[TweetRecord]) -> list[TradeSignalPair]:
    """
    Reconstructs completed trades from a list of classified tweets.

    The reconstruction rules are intentionally simple and deterministic:

    - Tweets are processed in chronological order.
    - Each symbol is tracked independently.
    - The first `bullish` tweet for a symbol opens a `long` trade.
    - The first `bearish` tweet for a symbol opens a `short` trade.
    - Repeated tweets with the same sentiment for an already-open symbol are
      ignored. For example, three consecutive `bullish` tweets on `AAPL` still
      represent only one open long position.
    - When the opposite sentiment appears on the same symbol, the currently
      open trade is closed and emitted as a `TradeSignalPair`.
    - That same opposite-sentiment tweet immediately becomes the opening tweet
      for the next position in the opposite direction. This makes sequences
      like `bullish -> bearish -> bullish` produce one completed long trade and
      leave the final bullish tweet as the opening signal for a new position.
    - Any trade still open at the end of the tweet history is intentionally
      excluded from the returned output because it has no closing signal.

    The function first filters to actionable tweets and sorts them
    chronologically so the result is deterministic even if the caller provides
    unsorted input.
    """

    trade_tweets = filter_trade_tweets(tweets)
    completed_trades: list[TradeSignalPair] = []

    # Track the currently open signal per symbol. The stored value is the
    # opening tweet for that symbol's active position.
    open_positions: dict[str, TweetRecord] = {}

    for tweet in trade_tweets:
        symbol = tweet.symbol
        if symbol is None:
            continue

        open_tweet = open_positions.get(symbol)

        # No position is open yet for this symbol, so this tweet becomes the
        # opening signal.
        if open_tweet is None:
            open_positions[symbol] = tweet
            continue

        # Repeated sentiment does not change the existing position.
        if tweet.sentiment == open_tweet.sentiment:
            continue

        direction = "long" if open_tweet.sentiment == "bullish" else "short"

        # Opposite sentiment closes the current trade.
        completed_trades.append(
            TradeSignalPair(
                symbol=symbol,
                direction=direction,
                entry_tweet=open_tweet,
                exit_tweet=tweet,
                entry_time=open_tweet.created_at,
                exit_time=tweet.created_at,
            )
        )

        # The closing tweet also opens the next trade in the opposite direction.
        open_positions[symbol] = tweet

    return completed_trades
