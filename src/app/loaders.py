
"""
Load and normalize raw tweet and price data for the app.

This module is responsible for reading the source files used by the project
and converting them into simple, consistent data structures. It keeps the
loading layer separate from any analysis, trade evaluation, or symbol logic.
"""

import json
import os
from datetime import datetime

import pandas as pd


PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _normalize_tweets(raw_tweets: list[dict]) -> list[dict]:
    """Normalize raw tweet payloads into the fields the app needs."""

    normalized_tweets = []
    for tweet in raw_tweets:
        # The source data uses a trailing "Z" for UTC, so convert it to an
        # ISO 8601 offset that `datetime.fromisoformat` can parse directly.
        timestamp = tweet["timestamp"].replace("Z", "+00:00")
        normalized_tweets.append(
            {
                "text": tweet.get("text", "").strip(),
                "created_at": datetime.fromisoformat(timestamp),
            }
        )

    return normalized_tweets


def load_tweets(file_path: str) -> list[dict]:
    """Load tweets from JSON and keep only the normalized fields the app needs."""
    with open(file_path, "r", encoding="utf-8") as file:
        tweets = json.load(file)

    return _normalize_tweets(tweets)


def load_tweets_from_file_like(file_obj) -> list[dict]:
    """Load tweets from a file-like object and normalize them like disk input."""

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    tweets = json.load(file_obj)
    return _normalize_tweets(tweets)


def load_price_data(
    folder_path: str,
    symbols: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load price CSV files in a folder into a symbol-to-DataFrame mapping.

    When `symbols` is provided, only matching CSV files are loaded. When it is
    `None`, all price CSVs in the folder are loaded.
    """

    price_data = {}
    # Normalize requested symbols so file matching is case-insensitive
    requested_symbols = {symbol.upper() for symbol in symbols} if symbols else None

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".csv"):
            continue

        symbol = os.path.splitext(filename)[0].upper()
        if requested_symbols is not None and symbol not in requested_symbols:
            continue

        file_path = os.path.join(folder_path, filename)

        # These project CSVs are headerless, so we assign the expected OHLCV
        # column names while reading them in.
        df = pd.read_csv(file_path, header=None, names=PRICE_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        price_data[symbol] = df

    return price_data
