
"""
LLM-based tweet classification helpers for trading signal extraction.

This module sends tweets to an LLM and asks for a strict JSON classification
result. The model decides whether the tweet is a usable "bullish", "bearish",
or "neutral" trading signal and, when possible, extracts one main symbol.

The implementation is intentionally small and conservative:
- batched LLM calls for likely candidates
- optional concurrent execution for better UI responsiveness
- strict JSON output
- simple parsing and validation
- neutral fallback on any failure
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from openai import OpenAI

from .models import TweetRecord


# Keyword sets used to reduce unnecessary LLM calls
TRADE_KEYWORDS = {
    "add",
    "added",
    "adding",
    "above resistance",
    "bearish",
    "below support",
    "bought",
    "bounce",
    "break down",
    "break out",
    "breaking down",
    "breaking out",
    "breakdown",
    "breakout",
    "bullish",
    "buy",
    "calls",
    "fade",
    "going long",
    "going short",
    "long",
    "pattern",
    "possible",
    "puts",
    "rejection",
    "resistance",
    "resistance holding",
    "sell",
    "short",
    "starter long",
    "starter short",
    "setup",
    "support",
    "support holding",
    "target",
    "watching",
    "interesting",
    "maybe",
}

SETUP_KEYWORDS = {
    "interesting",
    "maybe",
    "pattern",
    "possible",
    "setup",
    "watching",
}

# Patterns used for fast candidate detection before LLM calls
CASH_TAG_PATTERN = re.compile(r"\$[A-Z]{1,6}\b")
FOREX_TAG_PATTERN = re.compile(r"#[A-Z]{2,6}/[A-Z]{2,6}\b")
UPPERCASE_TOKEN_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")
COMMON_NON_TICKERS = {"USD", "EUR", "GBP", "AUD", "CAD", "JPY"}
DEFAULT_CACHE_PATH = ".cache/classifications.json"
BULLISH_RULE_INDICATORS = {
    "add",
    "added",
    "adding",
    "above resistance",
    "bought",
    "bounce",
    "break out",
    "breaking out",
    "breakout",
    "bullish",
    "buy",
    "calls",
    "going long",
    "long",
    "over resistance",
    "starter long",
    "support holding",
}
BEARISH_RULE_INDICATORS = {
    "bearish",
    "below support",
    "break down",
    "breaking down",
    "breakdown",
    "fade",
    "going short",
    "puts",
    "rejection",
    "resistance holding",
    "sell",
    "short",
    "sold",
    "starter short",
    "under support",
}
AMBIGUOUS_RULE_TERMS = {"cover", "covered", "covering"}


def _default_classification() -> dict:
    """Returns the safe fallback classification used on any failure."""

    return {"sentiment": "neutral", "symbol": None}


def _get_required_env(name: str) -> str:
    """Reads a required environment variable and raises if it is missing."""

    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_tweet_cache_key(tweet_text: str) -> str:
    """Returns a stable hash key for caching one tweet's classification."""

    return hashlib.sha256(tweet_text.encode("utf-8")).hexdigest()


def load_classification_cache(cache_path: str) -> dict:
    """
    Loads the local tweet classification cache from disk.

    Missing or corrupted cache files are treated as empty caches so the app can
    continue without failing.
    """

    try:
        with open(cache_path, "r", encoding="utf-8") as file:
            cache = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}

    if not isinstance(cache, dict):
        return {}

    normalized_cache = {}
    for cache_key, payload in cache.items():
        if not isinstance(cache_key, str) or not isinstance(payload, dict):
            continue
        normalized_cache[cache_key] = _normalize_classification_payload(payload)

    return normalized_cache


def save_classification_cache(cache: dict, cache_path: str) -> None:
    """
    Saves the local tweet classification cache to disk.

    Cache save failures are ignored so a filesystem issue does not crash the
    full analysis run.
    """

    try:
        cache_folder = os.path.dirname(cache_path)
        if cache_folder:
            os.makedirs(cache_folder, exist_ok=True)

        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(cache, file, indent=2, sort_keys=True)
    except OSError:
        return


def build_classification_prompt(tweet_text: str) -> str:
    """Builds a concise prompt asking the model for strict JSON output."""

    return f"""
Classify this tweet as one of:
- "bullish": a usable bullish trade call
- "bearish": a usable bearish trade call
- "neutral": not a usable trade call

Also extract the one main symbol only if the tweet is bullish or bearish and
the symbol is clear. If the tweet is neutral, or the symbol is unclear, return
"symbol": null.

Do not infer symbols aggressively.

Return only valid JSON in exactly this shape:
{{
  "sentiment": "bullish",
  "symbol": "AAPL"
}}

Valid sentiment values are only:
"bullish"
"bearish"
"neutral"

Tweet:
{tweet_text}
""".strip()


def _normalize_classification_payload(payload: dict) -> dict:
    """Validates a parsed classification payload and normalizes the symbol."""

    sentiment = payload.get("sentiment")
    symbol = payload.get("symbol")

    if sentiment not in {"bullish", "bearish", "neutral"}:
        return _default_classification()

    if symbol == "":
        symbol = None

    if symbol is not None and not isinstance(symbol, str):
        return _default_classification()

    if isinstance(symbol, str):
        symbol = symbol.strip().upper() or None

    # Neutral tweets should never carry a symbol into reconstruction
    if sentiment == "neutral":
        symbol = None

    return {"sentiment": sentiment, "symbol": symbol}


def parse_classification_response(response_text: str) -> dict:
    """Parses and validates the model response, falling back on malformed data."""

    try:
        payload = json.loads(response_text)
    except (TypeError, json.JSONDecodeError):
        start = response_text.find("{") if isinstance(response_text, str) else -1
        end = response_text.rfind("}") if isinstance(response_text, str) else -1

        # Some model responses include extra wrapper text around the JSON payload.
        if start == -1 or end == -1 or start >= end:
            return _default_classification()

        try:
            payload = json.loads(response_text[start : end + 1])
        except json.JSONDecodeError:
            return _default_classification()

    if not isinstance(payload, dict):
        return _default_classification()

    return _normalize_classification_payload(payload)


def call_llm(prompt: str) -> str:
    """Sends a single prompt to the configured LLM and returns the raw text."""

    api_key = _get_required_env("LLM_API_KEY")
    model = _get_required_env("LLM_MODEL")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify tweets into trading sentiment and extract a "
                    "single symbol. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content or ""


def classify_tweet(tweet_text: str) -> dict:
    """Classifies one tweet with the LLM and returns a validated result."""

    prompt = build_classification_prompt(tweet_text)

    try:
        response_text = call_llm(prompt)
        return parse_classification_response(response_text)
    except Exception:
        # Fail closed to neutral so one API/parsing issue does not stop the full app
        return _default_classification()


def _contains_any_indicator(text: str, indicators: set[str]) -> bool:
    """Returns True when the tweet contains any keyword or phrase indicator."""

    return any(indicator in text for indicator in indicators)


def is_likely_trade_candidate(tweet_text: str) -> bool:
    """
    Quickly checks whether a tweet looks like it may contain a trade call.

    This is only a responsiveness optimization to reduce unnecessary LLM calls.
    It is intentionally lightweight and conservative, and should not be treated
    as the final classification logic.
    """

    if not tweet_text:
        return False

    lowercase_text = tweet_text.lower()
    has_trade_or_setup_language = _contains_any_indicator(
        lowercase_text,
        TRADE_KEYWORDS | SETUP_KEYWORDS,
    )

    if not has_trade_or_setup_language:
        return False

    if CASH_TAG_PATTERN.search(tweet_text):
        return True

    if FOREX_TAG_PATTERN.search(tweet_text):
        return True

    tokens = UPPERCASE_TOKEN_PATTERN.findall(tweet_text)
    has_ticker_like_token = any(token not in COMMON_NON_TICKERS for token in tokens)
    return has_ticker_like_token


def _extract_rule_symbols(tweet_text: str, has_trade_language: bool) -> list[str]:
    """Extracts a small set of confident symbols for rule-based classification."""

    cashtags = [match.upper() for match in CASH_TAG_PATTERN.findall(tweet_text)]
    if cashtags:
        return list(dict.fromkeys(symbol.lstrip("$") for symbol in cashtags))

    forex_tags = [match.upper() for match in FOREX_TAG_PATTERN.findall(tweet_text)]
    if forex_tags:
        return list(dict.fromkeys(symbol.lstrip("#") for symbol in forex_tags))

    if not has_trade_language:
        return []

    uppercase_tokens = [
        token
        for token in UPPERCASE_TOKEN_PATTERN.findall(tweet_text)
        if token not in COMMON_NON_TICKERS
    ]
    return list(dict.fromkeys(uppercase_tokens))


def classify_tweet_with_rules(tweet_text: str) -> dict | None:
    """
    Applies simple rule-based classification before falling back to the LLM.

    This is a cold-start optimization for obvious trade calls and obvious
    non-trade tweets. The LLM is still reserved for ambiguous likely trade
    calls that the rules cannot classify confidently.
    """

    if not tweet_text:
        return {"sentiment": "neutral", "symbol": None}

    lowercase_text = tweet_text.lower()
    has_bullish_signal = _contains_any_indicator(lowercase_text, BULLISH_RULE_INDICATORS)
    has_bearish_signal = _contains_any_indicator(lowercase_text, BEARISH_RULE_INDICATORS)
    has_trade_language = has_bullish_signal or has_bearish_signal
    has_setup_language = _contains_any_indicator(lowercase_text, SETUP_KEYWORDS)
    has_symbol_evidence = bool(
        CASH_TAG_PATTERN.search(tweet_text)
        or FOREX_TAG_PATTERN.search(tweet_text)
        or UPPERCASE_TOKEN_PATTERN.search(tweet_text)
    )

    if _contains_any_indicator(lowercase_text, AMBIGUOUS_RULE_TERMS):
        return None

    symbols = _extract_rule_symbols(tweet_text, has_trade_language)

    if not symbols:
        return {"sentiment": "neutral", "symbol": None}

    if not has_bullish_signal and not has_bearish_signal:
        if has_setup_language or has_symbol_evidence:
            return None
        return {"sentiment": "neutral", "symbol": None}

    if len(symbols) != 1:
        return None

    if has_bullish_signal and has_bearish_signal:
        return None

    if has_bullish_signal:
        return {"sentiment": "bullish", "symbol": symbols[0]}

    if has_bearish_signal:
        return {"sentiment": "bearish", "symbol": symbols[0]}

    return None


def build_batch_classification_prompt(tweet_texts: list[str]) -> str:
    """Builds a strict JSON prompt for classifying a batch of tweets."""

    lines = []
    for index, tweet_text in enumerate(tweet_texts):
        lines.append(f"{index}. {tweet_text}")

    tweets_block = "\n".join(lines)

    return f"""
Classify each tweet below as one of:
- "bullish": a usable bullish trade call
- "bearish": a usable bearish trade call
- "neutral": not a usable trade call

Also extract the one main symbol only if the tweet is bullish or bearish and
the symbol is clear. If the tweet is neutral, or the symbol is unclear, return
"symbol": null.

Do not infer symbols aggressively.

Return only valid JSON in exactly this shape:
{{
  "results": [
    {{"index": 0, "sentiment": "bullish", "symbol": "AAPL"}},
    {{"index": 1, "sentiment": "neutral", "symbol": null}}
  ]
}}

Every input tweet must appear exactly once in "results" with the same index.

Tweets:
{tweets_block}
""".strip()


def classify_tweet_batch(tweet_texts: list[str]) -> list[dict]:
    """
    Classifies a batch of tweets in one LLM request.

    The response must include one result per input index. If the batch request
    fails or returns malformed data, the function falls back to classifying
    each tweet individually.
    """

    if not tweet_texts:
        return []

    prompt = build_batch_classification_prompt(tweet_texts)

    try:
        response_text = call_llm(prompt)
        payload = json.loads(response_text)

        if not isinstance(payload, dict):
            raise ValueError("Batch response must be a JSON object")

        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(tweet_texts):
            raise ValueError("Batch response must contain one result per tweet")

        ordered_results = [_default_classification() for _ in tweet_texts]
        seen_indexes = set()

        for item in results:
            if not isinstance(item, dict):
                raise ValueError("Batch result entries must be objects")

            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(tweet_texts):
                raise ValueError("Batch result index is invalid")

            if index in seen_indexes:
                raise ValueError("Batch result indexes must be unique")

            seen_indexes.add(index)
            ordered_results[index] = _normalize_classification_payload(item)

        if len(seen_indexes) != len(tweet_texts):
            raise ValueError("Batch response is missing indexes")

        return ordered_results
    except Exception:
        return [classify_tweet(tweet_text) for tweet_text in tweet_texts]


def annotate_tweets_with_llm(
    tweets: list[TweetRecord],
    max_workers: int = 4,
    batch_size: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> list[TweetRecord]:
    """
    Updates tweet records in place with LLM-based sentiment and symbol data.

    Tweets first pass through a lightweight pre-filter so obvious non-trade
    posts can be marked neutral without an API call. Cached classifications are
    reused first, then obvious trade calls are handled by a small rule-based
    classifier as a cold-start optimization. The LLM is reserved for ambiguous
    likely trade calls, which are classified in batches and processed with a
    small thread pool while still preserving the original tweet order. New
    results are saved to a local JSON cache as batches complete. If a batch
    fails, it falls back to per-tweet classification. When provided,
    `progress_callback` receives `(completed, total)` updates for unresolved
    LLM candidate batches as they finish.
    """

    if not tweets:
        return tweets

    worker_count = max(1, max_workers)
    batch_length = max(1, batch_size)
    cache = load_classification_cache(cache_path)
    uncached_candidate_tweets = []
    cache_changed = False
    cache_hits = 0
    rule_classified = 0
    neutral_non_candidates = 0
    llm_candidate_tweets = 0

    # Resolve each tweet through the cheapest reliable path first:
    # cache → deterministic rules → LLM batch classification.
    for tweet in tweets:
        cache_key = get_tweet_cache_key(tweet.text)
        cached_classification = cache.get(cache_key)
        if cached_classification is not None:
            tweet.sentiment = cached_classification["sentiment"]
            tweet.symbol = cached_classification["symbol"]
            cache_hits += 1
            continue

        rule_classification = classify_tweet_with_rules(tweet.text)
        if rule_classification is not None:
            tweet.sentiment = rule_classification["sentiment"]
            tweet.symbol = rule_classification["symbol"]
            cache[cache_key] = rule_classification
            cache_changed = True
            rule_classified += 1
            continue

        if not is_likely_trade_candidate(tweet.text):
            tweet.sentiment = "neutral"
            tweet.symbol = None
            neutral_non_candidates += 1
            continue

        uncached_candidate_tweets.append(tweet)
        llm_candidate_tweets += 1

    if cache_changed:
        save_classification_cache(cache, cache_path)

    if not uncached_candidate_tweets:
        if progress_callback:
            progress_callback(0, 0)
        print(
            "Classification summary: "
            f"cache={cache_hits}, rules={rule_classified}, "
            f"neutral={neutral_non_candidates}, llm={llm_candidate_tweets}"
        )
        return tweets

    candidate_batches = [
        uncached_candidate_tweets[index : index + batch_length]
        for index in range(0, len(uncached_candidate_tweets), batch_length)
    ]
    completed = 0
    total_batches = len(candidate_batches)
    classification_results = [None] * len(candidate_batches)

    if progress_callback:
        progress_callback(completed, total_batches)

    # Reserve the LLM for likely trade-call candidates so obvious non-signals
    # stay cheap and fast, while we still write the final results back in order.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                classify_tweet_batch,
                [tweet.text for tweet in batch],
            ): index
            for index, batch in enumerate(candidate_batches)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            classification_results[index] = future.result()
            completed += 1

            if progress_callback:
                progress_callback(completed, total_batches)

    for batch, batch_results in zip(candidate_batches, classification_results):
        for tweet, classification in zip(batch, batch_results):
            # Update the existing TweetRecord objects so later pipeline steps
            # see the results without rebuilding the list.
            tweet.sentiment = classification["sentiment"]
            tweet.symbol = classification["symbol"]
            cache[get_tweet_cache_key(tweet.text)] = classification
            cache_changed = True

    if cache_changed:
        save_classification_cache(cache, cache_path)

    print(
        "Classification summary: "
        f"cache={cache_hits}, rules={rule_classified}, "
        f"neutral={neutral_non_candidates}, llm={llm_candidate_tweets}"
    )

    return tweets
