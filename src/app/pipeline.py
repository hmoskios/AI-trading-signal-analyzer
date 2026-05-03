
"""
Simple end-to-end pipeline for the trading analysis app.

This module wires together the flow used by the backend and the UI:

1. load raw tweet data
2. convert tweets into `TweetRecord` objects
3. annotate tweets with LLM-based classification
4. reconstruct completed trade signal pairs
5. load only the price data needed for reconstructed symbols
6. score the reconstructed trades with price history
7. build a lightweight summary dictionary
"""

from collections.abc import Callable
from io import StringIO

from .classify import annotate_tweets_with_llm
from .loaders import load_price_data, load_tweets, load_tweets_from_file_like
from .models import TweetRecord
from .reconstruct import reconstruct_trades
from .score import build_summary, score_trades


def convert_to_records(raw_tweets: list[dict]) -> list[TweetRecord]:
    """
    Converts normalized tweet dictionaries into `TweetRecord` objects.

    The loader already standardizes each raw tweet into a dictionary with the
    keys the app needs. At this stage, `sentiment` and `symbol` are left as
    `None` because annotation happens in a separate LLM-based classification step.
    """

    tweet_records = []

    # Convert each normalized dict into a strongly-typed TweetRecord dataclass.
    for raw_tweet in raw_tweets:
        tweet_records.append(
            TweetRecord(
                text=raw_tweet["text"],
                created_at=raw_tweet["created_at"],
            )
        )

    return tweet_records


def run_pipeline(
    tweet_file: str,
    price_folder: str,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Runs the full tweet-to-trade scoring pipeline.

    The pipeline performs the current end-to-end app flow:
    - load tweet data from disk
    - convert tweets into `TweetRecord` objects
    - annotate tweets with LLM-based sentiment and symbol values
    - reconstruct completed trade signal pairs
    - load price data only for the reconstructed symbols
    - score completed trades with price data
    - build an aggregate summary

    Returns a dictionary containing the intermediate and final outputs that the
    UI can consume later. When provided, `status_callback` receives short stage
    updates as the pipeline moves through its main steps.
    """

    def emit_status(message: str) -> None:
        """Sends a stage update when the caller wants progress reporting."""

        if status_callback:
            status_callback(message)

    emit_status("Loading tweet data")
    raw_tweets = load_tweets(tweet_file)

    emit_status("Converting tweets")
    tweets = convert_to_records(raw_tweets)

    emit_status("Classifying tweets")
    annotate_tweets_with_llm(
        tweets,
        max_workers=4,
        batch_size=100,
        progress_callback=lambda completed, total: emit_status(
            f"Classified {completed} / {total} candidate batches"
        ),
    )

    emit_status("Reconstructing trades")
    trade_pairs = reconstruct_trades(tweets)

    emit_status("Loading price data")
    # Load price CSVs after reconstruction so we only read symbols that are
    # actually needed for the completed trade pairs.
    symbols = {trade_pair.symbol.upper() for trade_pair in trade_pairs if trade_pair.symbol}
    price_data = load_price_data(price_folder, symbols=symbols)

    emit_status("Scoring trades")
    trades = score_trades(trade_pairs, price_data)

    emit_status("Building summary")
    summary = build_summary(trades)

    return {
        "tweets": tweets,
        "trade_pairs": trade_pairs,
        "trades": trades,
        "summary": summary,
    }


def run_pipeline_from_uploaded_file(
    uploaded_file,
    price_folder: str,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Runs the same pipeline using tweets loaded from an uploaded JSON file.

    The uploaded file can be a text file-like object, raw bytes, or a string.
    Price data still comes from the included local folder on disk. When
    provided, `status_callback` receives short stage updates for UI progress.
    """

    def emit_status(message: str) -> None:
        """Sends a stage update when the caller wants progress reporting."""

        if status_callback:
            status_callback(message)

    if isinstance(uploaded_file, bytes):
        tweet_source = StringIO(uploaded_file.decode("utf-8"))
    elif isinstance(uploaded_file, str):
        tweet_source = StringIO(uploaded_file)
    else:
        tweet_source = uploaded_file

    emit_status("Loading tweet data")
    raw_tweets = load_tweets_from_file_like(tweet_source)

    emit_status("Converting tweets")
    tweets = convert_to_records(raw_tweets)

    emit_status("Classifying tweets")
    annotate_tweets_with_llm(
        tweets,
        max_workers=4,
        batch_size=100,
        progress_callback=lambda completed, total: emit_status(
            f"Classified {completed} / {total} candidate batches"
        ),
    )

    emit_status("Reconstructing trades")
    trade_pairs = reconstruct_trades(tweets)

    emit_status("Loading price data")
    # Load price CSVs after reconstruction so we only read symbols that are
    # actually needed for the completed trade pairs.
    symbols = {trade_pair.symbol.upper() for trade_pair in trade_pairs if trade_pair.symbol}
    price_data = load_price_data(price_folder, symbols=symbols)

    emit_status("Scoring trades")
    trades = score_trades(trade_pairs, price_data)

    emit_status("Building summary")
    summary = build_summary(trades)

    return {
        "tweets": tweets,
        "trade_pairs": trade_pairs,
        "trades": trades,
        "summary": summary,
    }
