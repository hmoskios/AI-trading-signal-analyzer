
"""
Tests for the simple end-to-end pipeline helpers.

These checks focus on the current behavior in `pipeline.py`, including raw
tweet conversion and the full pipeline orchestration from files on disk to
summary output.
"""

import json

from datetime import datetime

from src.app.pipeline import convert_to_records, run_pipeline


def test_convert_to_records_builds_tweet_records_without_annotations():
    """Converts raw tweet dictionaries into TweetRecord objects."""

    raw_tweets = [
        {
            "text": "Bullish AAPL",
            "created_at": datetime(2024, 1, 2, 9, 30),
        }
    ]

    tweets = convert_to_records(raw_tweets)

    assert len(tweets) == 1
    assert tweets[0].text == "Bullish AAPL"
    assert tweets[0].created_at == datetime(2024, 1, 2, 9, 30)
    assert tweets[0].sentiment is None
    assert tweets[0].symbol is None


def test_run_pipeline_uses_patched_annotation_step(tmp_path, monkeypatch):
    """
    Runs the pipeline with a patched annotation step instead of the LLM.

    The pipeline now passes explicit batching settings into the annotation
    helper, so this stub accepts the same arguments while still returning fully
    deterministic annotations for the test.
    """

    tweet_file = tmp_path / "tweets.json"
    price_folder = tmp_path / "prices"
    price_folder.mkdir()

    tweet_file.write_text(
        json.dumps(
            [
                {"text": "bullish AAPL", "timestamp": "2024-01-02T09:30:00Z"},
                {"text": "bearish AAPL", "timestamp": "2024-01-04T10:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )

    (price_folder / "AAPL.csv").write_text(
        "\n".join(
            [
                "2024-01-02,99,102,98,100,1000",
                "2024-01-03,101,104,100,103,1100",
                "2024-01-04,109,111,108,110,1200",
            ]
        ),
        encoding="utf-8",
    )

    def fake_annotate_tweets_with_llm(
        tweets,
        max_workers=4,
        batch_size=100,
        progress_callback=None,
    ):
        """Assign deterministic annotations and capture the batch settings."""

        assert max_workers == 4
        assert batch_size == 100
        assert progress_callback is not None

        tweets[0].sentiment = "bullish"
        tweets[0].symbol = "AAPL"
        tweets[1].sentiment = "bearish"
        tweets[1].symbol = "AAPL"
        return tweets

    monkeypatch.setattr(
        "src.app.pipeline.annotate_tweets_with_llm",
        fake_annotate_tweets_with_llm,
    )

    result = run_pipeline(str(tweet_file), str(price_folder))

    assert list(result.keys()) == ["tweets", "trade_pairs", "trades", "summary"]
    assert len(result["tweets"]) == 2
    assert result["tweets"][0].sentiment == "bullish"
    assert result["tweets"][0].symbol == "AAPL"
    assert result["tweets"][1].sentiment == "bearish"
    assert result["tweets"][1].symbol == "AAPL"
    assert len(result["trade_pairs"]) == 1
    assert len(result["trades"]) == 1
    assert result["summary"] == {
        "completed_trade_count": 1,
        "correct_trade_count": 1,
        "incorrect_trade_count": 0,
        "completed_trade_accuracy": 100.0,
        "average_return_pct": ((110.0 - 99.0) / 99.0) * 100,
        "average_hold_duration_days": 2.0208333333333335,
    }
