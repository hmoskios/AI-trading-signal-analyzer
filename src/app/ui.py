
"""
Streamlit interface for reviewing tweet-based trading analysis results.

This module focuses only on:
- collecting the tweet input from the reviewer
- calling the existing analysis pipeline
- formatting pipeline outputs for display
- rendering a small set of summary metrics and tables

The UI does not implement business logic, authentication, or classification.
Repeated runs are sped up by the persistent local classification cache used by
the classifier, not by caching the full pipeline output in Streamlit.
"""

import os

import pandas as pd
import streamlit as st

from .pipeline import run_pipeline, run_pipeline_from_uploaded_file


DEFAULT_TWEET_FILE = "data/TheShortBear-tweets.json"
DEFAULT_PRICE_FOLDER = "data/stock_price_data"
PIPELINE_STAGES = [
    "Loading tweet data",
    "Converting tweets",
    "Classifying tweets",
    "Reconstructing trades",
    "Loading price data",
    "Scoring trades",
    "Building summary",
]

UI_STYLES = """
<style>
div[data-testid="stFileUploader"] section {
    padding: 0.85rem 1rem;
    border-radius: 14px;
}

.app-subtitle {
    color: #5b6475;
    margin-top: -0.35rem;
    margin-bottom: 1.25rem;
}

.section-kicker {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 1.55rem;
    font-weight: 700;
    color: #1f355d;
    margin-top: 0.1rem;
    margin-bottom: 0.7rem;
}

.section-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #1f355d;
    margin-top: 0.3rem;
    margin-bottom: 0.2rem;
    padding-bottom: 0.18rem;
    border-bottom: 2px solid #2f6fed;
    display: inline-block;
}

.metric-card {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    min-height: 94px;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}

.metric-icon {
    width: 42px;
    height: 42px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 700;
    flex: 0 0 auto;
}

.metric-content {
    min-width: 0;
}

.metric-label {
    color: #667085;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}

.metric-value {
    color: #16233b;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}
</style>
"""

TABLE_HEADER_STYLES = [
    {
        "selector": "th",
        "props": [
            ("background-color", "#eaf2ff"),
            ("color", "#1f4b8f"),
            ("font-weight", "600"),
        ],
    }
]


def _shorten_text(text: str, max_length: int = 80) -> str:
    """
    Shortens long tweet text so trade tables stay compact and easy to scan.

    The trades table should include entry and exit tweet text, but clipping
    very long values keeps the page more readable during review.
    """

    if len(text) <= max_length:
        return text

    return f"{text[: max_length - 3]}..."


def render_section_header(title: str) -> None:
    """Renders a section title with a thin blue underline."""

    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def _render_metric_card(
    label: str,
    value: str | int,
    icon: str,
    icon_background: str,
    icon_color: str,
) -> None:
    """Renders a lightweight metric card with a dashboard-style layout."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon" style="background:{icon_background}; color:{icon_color};">{icon}</div>
            <div class="metric-content">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_table(dataframe: pd.DataFrame):
    """Applies lightweight header styling for display tables."""

    return dataframe.style.set_table_styles(TABLE_HEADER_STYLES)


def tweets_to_dataframe(tweets: list) -> pd.DataFrame:
    """
    Converts tweet records into a DataFrame for display in the UI.

    The pipeline returns tweet objects, so this helper extracts only the fields
    that are useful to a reviewer and presents them in a simple tabular format.
    """

    rows = []

    for tweet in tweets:
        rows.append(
            {
                "Created At": tweet.created_at,
                "Text": tweet.text,
                "Sentiment": tweet.sentiment,
                "Symbol": tweet.symbol,
            }
        )

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty and "Created At" in dataframe.columns:
        dataframe["Created At"] = pd.to_datetime(dataframe["Created At"])

    return dataframe


def trades_to_dataframe(trades: list) -> pd.DataFrame:
    """
    Converts scored trade records into a DataFrame for display in the UI.

    The output keeps the raw trade values intact, with only light rounding and
    datetime conversion to make the table easier to scan.
    """

    rows = []

    for trade in trades:
        rows.append(
            {
                "Symbol": trade.symbol,
                "Direction": trade.direction,
                "Entry Time": trade.entry_time,
                "Exit Time": trade.exit_time,
                "Entry Tweet": _shorten_text(trade.entry_tweet.text, max_length=48),
                "Exit Tweet": _shorten_text(trade.exit_tweet.text, max_length=48),
                "Entry Price": trade.entry_price,
                "Exit Price": trade.exit_price,
                "Realized Return (%)": trade.return_pct,
                "Hold Duration (Days)": trade.hold_duration,
                "Outcome": "Correct" if trade.is_correct else "Incorrect",
            }
        )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        return dataframe

    # Apply light formatting so the trade table stays readable.
    for column_name in ["Entry Time", "Exit Time"]:
        if column_name in dataframe.columns:
            dataframe[column_name] = pd.to_datetime(dataframe[column_name])

    for column_name in [
        "Entry Price",
        "Exit Price",
        "Realized Return (%)",
        "Hold Duration (Days)",
    ]:
        if column_name in dataframe.columns:
            dataframe[column_name] = dataframe[column_name].round(2)

    return dataframe


def _stage_progress_value(message: str) -> float:
    """Maps pipeline status text to a simple 0-1 progress value."""

    if message.startswith("Classified "):
        classified_text = message.removeprefix("Classified ")
        completed_text, _, total_text = classified_text.partition(" / ")

        try:
            completed = int(completed_text.strip())
            total = int(total_text.split(" ", 1)[0].strip())
        except ValueError:
            return 2 / len(PIPELINE_STAGES)

        if total <= 0:
            return 3 / len(PIPELINE_STAGES)

        stage_start = 2 / len(PIPELINE_STAGES)
        stage_width = 1 / len(PIPELINE_STAGES)
        return stage_start + (completed / total) * stage_width

    if message in PIPELINE_STAGES:
        return PIPELINE_STAGES.index(message) / len(PIPELINE_STAGES)

    return 0.0


def render_summary(summary: dict, tweets: list) -> None:
    """
    Renders the high-level trade summary using dashboard-style metric cards.

    The summary values come directly from the scoring layer and are lightly
    rounded for presentation.
    """

    total_tweets = len(tweets)
    bullish_tweet_count = sum(1 for tweet in tweets if tweet.sentiment == "bullish")
    bearish_tweet_count = sum(1 for tweet in tweets if tweet.sentiment == "bearish")
    neutral_tweet_count = sum(1 for tweet in tweets if tweet.sentiment == "neutral")

    first_row = st.columns(4)
    with first_row[0]:
        _render_metric_card("Total Tweets Loaded", total_tweets, "●", "#e8f1ff", "#2f6fed")
    with first_row[1]:
        _render_metric_card("Bullish Tweets", bullish_tweet_count, "↗", "#eaf8ef", "#16a34a")
    with first_row[2]:
        _render_metric_card("Bearish Tweets", bearish_tweet_count, "↘", "#fdecec", "#dc2626")
    with first_row[3]:
        _render_metric_card("Neutral Tweets", neutral_tweet_count, "—", "#f1f5f9", "#64748b")

    second_row = st.columns(4)
    with second_row[0]:
        _render_metric_card("Completed Trades", int(summary.get("completed_trade_count", 0)), "◔", "#f3ecff", "#7c3aed")
    with second_row[1]:
        _render_metric_card("Correct Trades", int(summary.get("correct_trade_count", 0)), "✓", "#eaf8ef", "#16a34a")
    with second_row[2]:
        _render_metric_card("Incorrect Trades", int(summary.get("incorrect_trade_count", 0)), "✕", "#fdecec", "#dc2626")
    with second_row[3]:
        _render_metric_card(
            "Completed Trade Accuracy",
            f"{summary.get('completed_trade_accuracy', 0.0):.1f}%",
            "◎",
            "#e8f1ff",
            "#2f6fed",
        )

    third_row = st.columns(2)
    with third_row[0]:
        _render_metric_card(
            "Average Realized Returns (%)",
            f"{summary.get('average_return_pct', 0.0):.2f}%",
            "%",
            "#fff4e5",
            "#d97706",
        )
    with third_row[1]:
        _render_metric_card(
            "Average Hold Duration (Days)",
            f"{summary.get('average_hold_duration_days', 0.0):.2f}",
            "◷",
            "#e8f1ff",
            "#2f6fed",
        )


def render_app() -> None:
    """
    Renders the full single-page Streamlit application.

    This function sets up the page, collects the optional tweet upload, runs
    the pipeline when requested, handles errors, and displays the summary plus
    the tweet and trade tables.
    """

    st.set_page_config(page_title="AI-Powered Trading Signal Analyzer", layout="wide")
    st.markdown(UI_STYLES, unsafe_allow_html=True)

    st.title("📈 AI-Powered Trading Signal Analyzer")
    st.markdown(
        '<div class="app-subtitle">Analyze a trader’s tweet history, reconstruct completed trades, and evaluate performance using historical price data.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("How to use this app")
    st.info(
        "Upload a tweet JSON file, then click Run Analysis.\n\n"
        "If no file is uploaded, the app will analyze the included default tweet file (data/TheShortBear-tweets.json).\n\n"
        "Price data is already included and loaded automatically from data/stock_price_data."
    )

    render_section_header("Inputs")
    st.write("")

    uploaded_tweet_file = st.file_uploader("Select Tweet JSON File", type=["json"])
    st.caption(
        "The app pre-filters tweets, classifies likely trade-call candidate tweets in "
        "batches with the LLM, and reuses cached classification results from local storage."
    )

    # Warn the user if no LLM config is missing, since classification will be less
    # meaningful without it. Rule-based classification still runs, but ambiguous 
    # tweets fall back to neutral without the LLM.
    if not os.getenv("LLM_API_KEY") or not os.getenv("LLM_MODEL"):
        st.warning(
            "No LLM configuration detected.\n\n"
            "The app will still run using deterministic rule-based classification, "
            "but ambiguous tweets will default to neutral.\n\n"
            "To enable full analysis, add your API key and model to a `.env` file:\n"
            "`LLM_API_KEY=your_openai_api_key_here`\n"
            "`LLM_MODEL=gpt-4o-mini`"
        )

    run_analysis = st.button("Run Analysis", type="primary")

    if not run_analysis:
        return

    if not os.path.exists(DEFAULT_PRICE_FOLDER):
        st.error(
            "Price data not found.\n\n"
            "1. Download from the README link\n"
            "2. Extract the folder\n"
            "3. Place it at: `data/stock_price_data/`\n\n"
            "The folder name and location must match exactly."
        )
        return
    
    try:
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        text_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0.0)
        status_panel = status_placeholder.status("Starting analysis...", expanded=True)

        def update_status(message: str) -> None:
            """Refreshes the visible UI status during a cache miss."""

            status_panel.update(label=message, state="running")
            text_placeholder.caption(message)
            progress_bar.progress(_stage_progress_value(message))

        text_placeholder.caption("Preparing analysis...")

        if uploaded_tweet_file is None:
            pipeline_result = run_pipeline(
                DEFAULT_TWEET_FILE,
                DEFAULT_PRICE_FOLDER,
                status_callback=update_status,
            )
        else:
            pipeline_result = run_pipeline_from_uploaded_file(
                uploaded_tweet_file.getvalue(),
                DEFAULT_PRICE_FOLDER,
                status_callback=update_status,
            )

        progress_bar.progress(1.0)
        status_placeholder.empty()
        progress_placeholder.empty()
        text_placeholder.empty()
    except Exception as exc:
        st.error(f"Analysis failed. Please check the tweet JSON file. Details: {exc}")
        return

    summary = pipeline_result.get("summary", {})
    tweets = pipeline_result.get("tweets", [])
    trades = pipeline_result.get("trades", [])

    tweets_df = tweets_to_dataframe(tweets)
    trades_df = trades_to_dataframe(trades)

    st.success(f"Analysis complete. Processed {len(tweets)} tweets and produced {len(trades)} completed trades.")

    st.divider()
    st.markdown('<div class="section-kicker">📊 Results</div>', unsafe_allow_html=True)

    render_section_header("Summary")
    render_summary(summary, tweets)
    st.write("")
    st.write("")

    render_section_header("Tweet Sentiment & Symbol Extraction")
    st.dataframe(
        _style_table(tweets_df),
        use_container_width=True,
        hide_index=True,
    )

    render_section_header("Completed Trades")
    if trades_df.empty:
        st.info("No completed trades were produced for the selected inputs.")
    else:
        st.dataframe(
            _style_table(trades_df),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Entry Tweet": st.column_config.TextColumn(width="medium"),
                "Exit Tweet": st.column_config.TextColumn(width="medium"),
            },
        )
