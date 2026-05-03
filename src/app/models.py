
"""
Lightweight data models used throughout the trading analysis app.

These are simple containers for passing structured data between different
stages of the pipeline (loading → classification → reconstruction → scoring).
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TweetRecord:
    """
    Represents a single tweet after loading and normalization.

    This is the core unit of input data for the system. Additional fields
    like sentiment and symbol are populated later in the pipeline after
    LLM classification and symbol extraction.
    """

    text: str
    created_at: datetime

    # Populated during later stages of the pipeline
    sentiment: str | None = None  # "bullish", "bearish", or "neutral"
    symbol: str | None = None     # Extracted ticker symbol (if applicable)


@dataclass
class TradeRecord:
    """
    Represents one completed round-trip trade derived from tweet signals.
    
    A trade is formed when:
    - a tweet expresses a position (bullish or bearish), and
    - a later tweet on the same symbol expresses the opposite sentiment (closing signal)

    The entry tweet opens the position, and the exit tweet closes it.
    Trade performance is evaluated using price data between these timestamps.
    """

    symbol: str
    direction: str  # "long" (bullish → bearish) or "short" (bearish → bullish)
    
    entry_tweet: TweetRecord
    exit_tweet: TweetRecord
    
    entry_time: datetime
    exit_time: datetime
    
    entry_price: float
    exit_price: float
    
    return_pct: float           # Realized return over the trade (%)
    hold_duration: float        # Duration of trade in days

    is_correct: bool            # Whether trade direction matched price movement
