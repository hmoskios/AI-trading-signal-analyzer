
"""
Tests for the LLM-based tweet classification helpers.

These checks focus on prompt construction, response parsing, safe fallback
behavior, and in-place tweet annotation without making real API calls.

Run with: `python -m pytest tests/test_classify.py`
"""

import json

from datetime import datetime

from src.app.classify import (
    annotate_tweets_with_llm,
    build_classification_prompt,
    classify_tweet,
    classify_tweet_batch,
    classify_tweet_with_rules,
    get_tweet_cache_key,
    is_likely_trade_candidate,
    load_classification_cache,
    parse_classification_response,
    save_classification_cache,
)
from src.app.models import TweetRecord


def make_tweet(text: str) -> TweetRecord:
    """Creates a simple TweetRecord for classification tests."""

    return TweetRecord(
        text=text,
        created_at=datetime(2024, 1, 2, 9, 30),
    )


def test_build_classification_prompt_includes_tweet_text_and_json_shape():
    """Includes the tweet text, valid labels, and expected JSON keys."""

    tweet_text = "Bullish on aapl after earnings"

    prompt = build_classification_prompt(tweet_text)

    assert tweet_text in prompt
    assert '"bullish"' in prompt
    assert '"bearish"' in prompt
    assert '"neutral"' in prompt
    assert '"sentiment"' in prompt
    assert '"symbol"' in prompt


def test_parse_classification_response_accepts_valid_json():
    """Parses valid JSON and normalizes the symbol to uppercase."""

    response_text = '{"sentiment": "bullish", "symbol": "aapl"}'
    result = parse_classification_response(response_text)
    assert result == {"sentiment": "bullish", "symbol": "AAPL"}


def test_parse_classification_response_treats_empty_symbol_as_none():
    """Turns an empty symbol string into None."""

    response_text = '{"sentiment": "bullish", "symbol": ""}'
    result = parse_classification_response(response_text)
    assert result == {"sentiment": "bullish", "symbol": None}


def test_parse_classification_response_forces_neutral_symbol_to_none():
    """Drops any symbol when the sentiment is neutral."""

    response_text = '{"sentiment": "neutral", "symbol": "TSLA"}'
    result = parse_classification_response(response_text)
    assert result == {"sentiment": "neutral", "symbol": None}


def test_parse_classification_response_handles_wrapper_text_around_json():
    """Extracts and parses the first JSON object from wrapper text."""

    response_text = (
        'Here is the result: {"sentiment": "bearish", "symbol": "tsla"}'
    )
    result = parse_classification_response(response_text)
    assert result == {"sentiment": "bearish", "symbol": "TSLA"}


def test_parse_classification_response_falls_back_on_invalid_sentiment():
    """Returns the neutral fallback for unsupported sentiment values."""

    response_text = '{"sentiment": "positive", "symbol": "AAPL"}'
    result = parse_classification_response(response_text)
    assert result == {"sentiment": "neutral", "symbol": None}


def test_parse_classification_response_falls_back_on_malformed_json():
    """Returns the neutral fallback when the response is malformed."""

    response_text = '{"sentiment": "bullish", "symbol": "AAPL"'
    result = parse_classification_response(response_text)
    assert result == {"sentiment": "neutral", "symbol": None}


def test_is_likely_trade_candidate_detects_trade_signals():
    """Returns True for obvious trade-call tweets."""

    assert is_likely_trade_candidate("$AAPL breaking out") is True
    assert is_likely_trade_candidate("#AUD/USD looks bearish") is True
    assert is_likely_trade_candidate("Bullish TSLA into earnings") is True
    assert is_likely_trade_candidate("short NVDA at resistance") is True
    assert is_likely_trade_candidate("$AAPL possible setup") is True


def test_is_likely_trade_candidate_rejects_obvious_non_trade_tweets():
    """Returns False for tweets with no sign of a trade call."""

    assert is_likely_trade_candidate("Good morning everyone") is False
    assert is_likely_trade_candidate("Thanks for the follow") is False
    assert is_likely_trade_candidate("") is False
    assert is_likely_trade_candidate("buy the dip") is False
    assert is_likely_trade_candidate("TSLA into earnings") is False
    assert is_likely_trade_candidate("$AAPL") is False
    assert is_likely_trade_candidate("possible breakout soon") is False


def test_classify_tweet_returns_parsed_result_when_llm_call_succeeds(monkeypatch):
    """Returns the parsed normalized result when the LLM call succeeds."""

    def fake_call_llm(prompt: str) -> str:
        """Returns a valid JSON classification response."""

        assert "AAPL" in prompt
        return '{"sentiment": "bullish", "symbol": "aapl"}'

    monkeypatch.setattr("src.app.classify.call_llm", fake_call_llm)
    result = classify_tweet("AAPL looks strong here")
    assert result == {"sentiment": "bullish", "symbol": "AAPL"}


def test_classify_tweet_falls_back_when_llm_call_raises(monkeypatch):
    """Returns the neutral fallback when the LLM call raises an error."""

    def fake_call_llm(prompt: str) -> str:
        """Raises to simulate an LLM failure."""

        raise RuntimeError("LLM failed")

    monkeypatch.setattr("src.app.classify.call_llm", fake_call_llm)
    result = classify_tweet("TSLA is a mess")
    assert result == {"sentiment": "neutral", "symbol": None}


def test_get_tweet_cache_key_is_stable():
    """Returns the same cache key for the same tweet text."""

    assert get_tweet_cache_key("Bullish AAPL") == get_tweet_cache_key("Bullish AAPL")
    assert get_tweet_cache_key("Bullish AAPL") != get_tweet_cache_key("Bearish AAPL")


def test_load_classification_cache_returns_empty_for_missing_file(tmp_path):
    """Returns an empty cache when the cache file does not exist."""

    cache = load_classification_cache(str(tmp_path / "missing.json"))

    assert cache == {}


def test_load_classification_cache_returns_empty_for_corrupt_file(tmp_path):
    """Returns an empty cache when the cache file is corrupted."""

    cache_path = tmp_path / "corrupt.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    cache = load_classification_cache(str(cache_path))

    assert cache == {}


def test_save_classification_cache_writes_json_file(tmp_path):
    """Writes the classification cache to disk as JSON."""

    cache_path = tmp_path / ".cache" / "classifications.json"
    cache = {
        "abc": {"sentiment": "bullish", "symbol": "AAPL"},
    }

    save_classification_cache(cache, str(cache_path))

    saved_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert saved_cache == cache


def test_classify_tweet_batch_returns_results_in_input_order(monkeypatch):
    """Parses batch results and preserves the input order."""

    def fake_call_llm(prompt: str) -> str:
        """Returns a valid JSON batch classification response."""

        assert "0. $AAPL breaking out" in prompt
        assert "1. bearish TSLA here" in prompt
        return (
            '{"results": ['
            '{"index": 1, "sentiment": "bearish", "symbol": "tsla"}, '
            '{"index": 0, "sentiment": "bullish", "symbol": "aapl"}'
            "]}"
        )

    monkeypatch.setattr("src.app.classify.call_llm", fake_call_llm)

    results = classify_tweet_batch(["$AAPL breaking out", "bearish TSLA here"])

    assert results == [
        {"sentiment": "bullish", "symbol": "AAPL"},
        {"sentiment": "bearish", "symbol": "TSLA"},
    ]


def test_classify_tweet_with_rules_handles_obvious_long_and_short_calls():
    """Returns rule-based sentiment for obvious single-symbol trade calls."""

    assert classify_tweet_with_rules("$AAPL breakout") == {
        "sentiment": "bullish",
        "symbol": "AAPL",
    }
    assert classify_tweet_with_rules("$TSLA short here") == {
        "sentiment": "bearish",
        "symbol": "TSLA",
    }


def test_classify_tweet_with_rules_handles_obvious_bearish_cashtag():
    """Returns bearish for an obvious single-symbol bearish trade call."""

    assert classify_tweet_with_rules("$NVDA breakdown below support") == {
        "sentiment": "bearish",
        "symbol": "NVDA",
    }


def test_classify_tweet_with_rules_returns_none_for_conflicting_signals():
    """Falls through when bullish and bearish signals conflict."""

    assert classify_tweet_with_rules("$AAPL long but also short setup") is None


def test_classify_tweet_with_rules_returns_none_for_unclear_direction():
    """Falls through when a symbol exists but the setup is directionally unclear."""

    assert classify_tweet_with_rules("$AAPL possible setup") is None


def test_classify_tweet_with_rules_returns_neutral_for_no_symbol_tweet():
    """Returns neutral without the LLM for obvious non-trade tweets."""

    assert classify_tweet_with_rules("Good morning everyone") == {
        "sentiment": "neutral",
        "symbol": None,
    }


def test_annotate_tweets_with_llm_updates_records_in_place(tmp_path, monkeypatch):
    """Updates existing TweetRecord objects and returns the same list."""

    tweets = [
        make_tweet("$AAPL possible setup"),
        make_tweet("$TSLA watching pattern"),
    ]

    batch_calls = []

    def fake_classify_tweet_batch(tweet_texts: list[str]) -> list[dict]:
        """Returns predictable batch classifications for candidate tweets."""

        batch_calls.append(tweet_texts)
        return [
            {"sentiment": "bullish", "symbol": "AAPL"},
            {"sentiment": "bearish", "symbol": "TSLA"},
        ]

    monkeypatch.setattr("src.app.classify.classify_tweet_batch", fake_classify_tweet_batch)

    returned_tweets = annotate_tweets_with_llm(
        tweets,
        batch_size=100,
        cache_path=str(tmp_path / "classifications.json"),
    )

    assert returned_tweets is tweets
    assert batch_calls == [["$AAPL possible setup", "$TSLA watching pattern"]]
    assert tweets[0].sentiment == "bullish"
    assert tweets[0].symbol == "AAPL"
    assert tweets[1].sentiment == "bearish"
    assert tweets[1].symbol == "TSLA"


def test_annotate_tweets_with_llm_skips_non_candidates(tmp_path, monkeypatch):
    """Marks obvious non-candidates neutral without calling the classifier."""

    tweets = [make_tweet("Good morning everyone")]

    def fake_classify_tweet_batch(tweet_texts: list[str]) -> list[dict]:
        """Fails if a non-candidate tweet reaches batch classification."""

        raise AssertionError("classify_tweet_batch should not be called")

    monkeypatch.setattr("src.app.classify.classify_tweet_batch", fake_classify_tweet_batch)

    returned_tweets = annotate_tweets_with_llm(
        tweets,
        cache_path=str(tmp_path / "classifications.json"),
    )

    assert returned_tweets is tweets
    assert tweets[0].sentiment == "neutral"
    assert tweets[0].symbol is None


def test_annotate_tweets_with_llm_uses_cached_classification(tmp_path, monkeypatch):
    """Uses the local cache for candidate tweets without calling the LLM."""

    tweets = [make_tweet("$AAPL breaking out")]
    cache_path = tmp_path / ".cache" / "classifications.json"
    cache = {
        get_tweet_cache_key("$AAPL breaking out"): {
            "sentiment": "bullish",
            "symbol": "AAPL",
        }
    }
    save_classification_cache(cache, str(cache_path))

    def fake_classify_tweet_batch(tweet_texts: list[str]) -> list[dict]:
        """Fails if a cached tweet reaches batch classification."""

        raise AssertionError("classify_tweet_batch should not be called")

    monkeypatch.setattr("src.app.classify.classify_tweet_batch", fake_classify_tweet_batch)

    returned_tweets = annotate_tweets_with_llm(tweets, cache_path=str(cache_path))

    assert returned_tweets is tweets
    assert tweets[0].sentiment == "bullish"
    assert tweets[0].symbol == "AAPL"


def test_annotate_tweets_with_llm_uses_rules_without_calling_llm(tmp_path, monkeypatch):
    """Uses rule-based classification for obvious trade calls."""

    tweets = [
        make_tweet("$AAPL breakout"),
        make_tweet("$TSLA short here"),
    ]

    def fake_classify_tweet_batch(tweet_texts: list[str]) -> list[dict]:
        """Fails if a rule-resolved tweet reaches batch classification."""

        raise AssertionError("classify_tweet_batch should not be called")

    monkeypatch.setattr("src.app.classify.classify_tweet_batch", fake_classify_tweet_batch)

    returned_tweets = annotate_tweets_with_llm(
        tweets,
        cache_path=str(tmp_path / "classifications.json"),
    )

    assert returned_tweets is tweets
    assert tweets[0].sentiment == "bullish"
    assert tweets[0].symbol == "AAPL"
    assert tweets[1].sentiment == "bearish"
    assert tweets[1].symbol == "TSLA"


def test_annotate_tweets_with_llm_uses_llm_for_ambiguous_trade_calls(tmp_path, monkeypatch):
    """Falls through to batch classification when rules are ambiguous."""

    tweets = [make_tweet("$AAPL possible setup")]
    calls = []

    def fake_classify_tweet_batch(tweet_texts: list[str]) -> list[dict]:
        """Returns a result only for ambiguous tweets that reach the LLM path."""

        calls.append(tweet_texts)
        return [{"sentiment": "neutral", "symbol": None}]

    monkeypatch.setattr("src.app.classify.classify_tweet_batch", fake_classify_tweet_batch)

    returned_tweets = annotate_tweets_with_llm(
        tweets,
        cache_path=str(tmp_path / "classifications.json"),
    )

    assert returned_tweets is tweets
    assert tweets[0].sentiment == "neutral"
    assert tweets[0].symbol is None
    assert calls == [["$AAPL possible setup"]]


def test_annotate_tweets_with_llm_classifies_candidates_in_batches(tmp_path, monkeypatch):
    """Batches candidate tweets and updates records in place."""

    tweets = [
        make_tweet("Good morning everyone"),
        make_tweet("$AAPL possible setup"),
        make_tweet("$NVDA watching setup"),
    ]
    calls = []

    def fake_classify_tweet_batch(tweet_texts: list[str]) -> list[dict]:
        """Returns batch results only for candidate tweets."""

        calls.append(tweet_texts)
        assert tweet_texts == ["$AAPL possible setup", "$NVDA watching setup"]
        return [
            {"sentiment": "bullish", "symbol": "AAPL"},
            {"sentiment": "bearish", "symbol": "NVDA"},
        ]

    monkeypatch.setattr("src.app.classify.classify_tweet_batch", fake_classify_tweet_batch)

    returned_tweets = annotate_tweets_with_llm(
        tweets,
        batch_size=100,
        cache_path=str(tmp_path / "classifications.json"),
    )

    assert returned_tweets is tweets
    assert tweets[0].sentiment == "neutral"
    assert tweets[0].symbol is None
    assert tweets[1].sentiment == "bullish"
    assert tweets[1].symbol == "AAPL"
    assert tweets[2].sentiment == "bearish"
    assert tweets[2].symbol == "NVDA"
    assert calls == [["$AAPL possible setup", "$NVDA watching setup"]]


def test_parse_classification_response_rejects_non_string_symbol():
    """Falls back when symbol is not a string."""

    response_text = '{"sentiment": "bullish", "symbol": 123}'
    result = parse_classification_response(response_text)
    assert result == {"sentiment": "neutral", "symbol": None}
