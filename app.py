import streamlit as st
import pandas as pd
import plotly.express as px

from sentiment_analysis import (
    fetch_news,
    analyze_article,
    load_articles_from_cache
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Gold Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📈 Gold Market Sentiment Analyzer")

st.markdown(
    """
    Analyze financial news using **FinBERT** and **VADER**
    to estimate the overall sentiment of the gold market.
    """
)

st.divider()


# =========================================================
# SETTINGS
# =========================================================

st.sidebar.header("Analysis Settings")

num_articles = st.sidebar.slider(
    "Articles per query",
    min_value=5,
    max_value=20,
    value=10
)

use_cache = st.sidebar.checkbox(
    "Use cached news if live fetching fails",
    value=True
)

run_analysis = st.sidebar.button(
    "🔄 Run Analysis",
    type="primary"
)


# =========================================================
# FUNCTIONS
# =========================================================

def calculate_finbert_score(results):
    """
    Convert FinBERT sentiment into a numerical score.

    Positive = +1
    Neutral  =  0
    Negative = -1
    """

    if not results:
        return 0.0

    score = 0

    for result in results:

        sentiment = result["finbert_sentiment"]

        if sentiment == "Positive":
            score += 1

        elif sentiment == "Negative":
            score -= 1

    return score / len(results)


def get_market_condition(score):

    if score > 0.20:
        return "Bullish"

    elif score < -0.20:
        return "Bearish"

    else:
        return "Neutral"


def get_sentiment_counts(results, model):

    counts = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    for result in results:

        if model == "FinBERT":
            sentiment = result["finbert_sentiment"]

        else:
            sentiment = result["vader_sentiment"]

        counts[sentiment] += 1

    return counts


# =========================================================
# ANALYSIS
# =========================================================

if run_analysis:

    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment"
    ]

    all_articles = []

    progress = st.progress(0)

    status = st.empty()

    try:

        # -------------------------------------------------
        # FETCH LIVE NEWS
        # -------------------------------------------------

        for index, query in enumerate(queries):

            status.write(
                f"Fetching news for: **{query}**"
            )

            articles = fetch_news(
                query,
                num_articles
            )

            all_articles.extend(articles)

            progress.progress(
                (index + 1) / len(queries)
            )

        # -------------------------------------------------
        # REMOVE DUPLICATES
        # -------------------------------------------------

        unique_articles = {}

        for article in all_articles:

            link = article.get("link")

            if link:
                unique_articles[link] = article

        all_articles = list(
            unique_articles.values()
        )

    except Exception as e:

        st.warning(
            f"Live news fetching failed: {e}"
        )

        if use_cache:

            st.info(
                "Trying to use the local news cache..."
            )

            all_articles = load_articles_from_cache()

        else:

            all_articles = []

    progress.empty()
    status.empty()


    # -----------------------------------------------------
    # CHECK ARTICLES
    # -----------------------------------------------------

    if not all_articles and use_cache:

        st.warning(
            "Live news could not be retrieved. "
            "Trying the local news cache..."
        )

        all_articles = load_articles_from_cache()


    if not all_articles:

        st.error(
            "No news articles are available from "
            "either the live source or local cache."
        )

        st.stop()


    st.success(
        f"Successfully collected {len(all_articles)} unique articles."
    )


    # -----------------------------------------------------
    # ANALYZE ARTICLES
    # -----------------------------------------------------

    results = []

    analysis_progress = st.progress(0)

    for index, article in enumerate(all_articles):

        try:

            analysis = analyze_article(article)

            # Store article information along with analysis
            analysis["title"] = article["title"]

            analysis["link"] = article["link"]

            analysis["published"] = article.get(
                "published",
                "Unknown"
            )

            results.append(analysis)

        except Exception as e:

            st.warning(
                f"Could not analyze article: {article['title']}"
            )

        analysis_progress.progress(
            (index + 1) / len(all_articles)
        )

    analysis_progress.empty()


    # =====================================================
    # CALCULATE RESULTS
    # =====================================================

    total_articles = len(results)

    finbert_counts = get_sentiment_counts(
        results,
        "FinBERT"
    )

    vader_counts = get_sentiment_counts(
        results,
        "VADER"
    )


    # -----------------------------------------------------
    # FINBERT SCORE
    # -----------------------------------------------------

    finbert_score = calculate_finbert_score(
        results
    )

    market_condition = get_market_condition(
        finbert_score
    )


    if finbert_score > 0:

        overall_sentiment = "Positive"

    elif finbert_score < 0:

        overall_sentiment = "Negative"

    else:

        overall_sentiment = "Neutral"


    # -----------------------------------------------------
    # MODEL AGREEMENT
    # -----------------------------------------------------

    agreement_count = 0

    for result in results:

        if (
            result["finbert_sentiment"]
            ==
            result["vader_sentiment"]
        ):

            agreement_count += 1


    agreement_percentage = (
        agreement_count / total_articles * 100
        if total_articles > 0
        else 0
    )


    # =====================================================
    # DASHBOARD METRICS
    # =====================================================

    st.subheader("📊 Market Overview")

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Articles Analyzed",
        total_articles
    )


    col2.metric(
        "FinBERT Score",
        f"{finbert_score:+.2f}"
    )


    col3.metric(
        "Overall Sentiment",
        overall_sentiment
    )


    col4.metric(
        "Market Condition",
        market_condition
    )


    st.divider()


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.subheader("🤖 Model Comparison")

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # FINBERT
    # -----------------------------------------------------

    with col1:

        st.markdown("### FinBERT")

        finbert_df = pd.DataFrame(
            {
                "Sentiment": list(
                    finbert_counts.keys()
                ),
                "Articles": list(
                    finbert_counts.values()
                )
            }
        )


        fig_finbert = px.pie(
            finbert_df,
            names="Sentiment",
            values="Articles",
            title="FinBERT Sentiment Distribution",
            hole=0.4
        )


        st.plotly_chart(
            fig_finbert,
            use_container_width=True
        )


    # -----------------------------------------------------
    # VADER
    # -----------------------------------------------------

    with col2:

        st.markdown("### VADER")

        vader_df = pd.DataFrame(
            {
                "Sentiment": list(
                    vader_counts.keys()
                ),
                "Articles": list(
                    vader_counts.values()
                )
            }
        )


        fig_vader = px.pie(
            vader_df,
            names="Sentiment",
            values="Articles",
            title="VADER Sentiment Distribution",
            hole=0.4
        )


        st.plotly_chart(
            fig_vader,
            use_container_width=True
        )


    # =====================================================
    # MODEL AGREEMENT
    # =====================================================

    st.subheader("🔍 Model Agreement")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Agreement",
            f"{agreement_percentage:.2f}%"
        )


    with col2:

        st.metric(
            "Disagreement",
            f"{100 - agreement_percentage:.2f}%"
        )


    # =====================================================
    # MARKET INTERPRETATION
    # =====================================================

    st.subheader("📈 Overall Gold Market Sentiment")


    # -----------------------------------------------------
    # CALCULATE SENTIMENT PERCENTAGES
    # -----------------------------------------------------

    positive_percentage = (
        finbert_counts["Positive"]
        / total_articles
        * 100
        if total_articles > 0
        else 0
    )


    negative_percentage = (
        finbert_counts["Negative"]
        / total_articles
        * 100
        if total_articles > 0
        else 0
    )


    neutral_percentage = (
        finbert_counts["Neutral"]
        / total_articles
        * 100
        if total_articles > 0
        else 0
    )


    # -----------------------------------------------------
    # SENTIMENT METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Positive",
            f"{positive_percentage:.2f}%"
        )


    with col2:

        st.metric(
            "Negative",
            f"{negative_percentage:.2f}%"
        )


    with col3:

        st.metric(
            "Neutral",
            f"{neutral_percentage:.2f}%"
        )


    st.divider()


    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    st.write(
        f"### FinBERT Sentiment Score: "
        f"{finbert_score:+.2f}"
    )


    st.write(
        f"**Overall Sentiment:** "
        f"{overall_sentiment}"
    )


    st.write(
        f"**Market Classification:** "
        f"{market_condition.upper()}"
    )


    # -----------------------------------------------------
    # SENTIMENT INTERPRETATION
    # -----------------------------------------------------

    if finbert_score >= 0.20:

        interpretation = (
            "The analyzed financial headlines show a "
            "mildly positive sentiment toward the gold market."
        )


    elif finbert_score > 0:

        interpretation = (
            "The analyzed financial headlines show a "
            "slightly positive sentiment toward the gold market."
        )


    elif finbert_score <= -0.20:

        interpretation = (
            "The analyzed financial headlines show a "
            "negative sentiment toward the gold market."
        )


    elif finbert_score < 0:

        interpretation = (
            "The analyzed financial headlines show a "
            "slightly negative sentiment toward the gold market."
        )


    else:

        interpretation = (
            "The analyzed financial headlines show a "
            "neutral sentiment toward the gold market."
        )


    # -----------------------------------------------------
    # INTERPRETATION BOX
    # -----------------------------------------------------

    st.info(
        f"📌 **Market Interpretation**\n\n"
        f"FinBERT analyzed **{total_articles} financial news headlines**. "
        f"The aggregated sentiment score is **{finbert_score:+.2f}**, "
        f"indicating the current sentiment reflected in the analyzed "
        f"headlines.\n\n"
        f"**Interpretation:** {interpretation}\n\n"
        f"**Market Classification:** "
        f"**{market_condition.upper()}**"
    )


    # -----------------------------------------------------
    # SENTIMENT SCORE BAR
    # -----------------------------------------------------

    st.progress(
        min(
            max(
                (finbert_score + 1) / 2,
                0
            ),
            1
        )
    )


    st.caption(
        "Score range: -1.00 (Extremely Negative) "
        "to +1.00 (Extremely Positive)"
    )


    # =====================================================
    # ARTICLE TABLE
    # =====================================================

    st.divider()

    st.subheader("📰 Analyzed Financial News")

    table_data = []


    for result in results:

        table_data.append(
            {
                "Title": result["title"],

                "Published": result["published"],

                "FinBERT": result["finbert_sentiment"],

                "Confidence": (
                    f"{result['finbert_confidence'] * 100:.2f}%"
                ),

                "VADER": result["vader_sentiment"],

                "Polarity": (
                    f"{result['vader_polarity']:.2f}"
                ),

                "Agreement": (
                    "YES"
                    if result["finbert_sentiment"]
                    ==
                    result["vader_sentiment"]
                    else "NO"
                )
            }
        )


    articles_df = pd.DataFrame(
        table_data
    )


    st.dataframe(
        articles_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # DOWNLOAD RESULTS
    # =====================================================

    csv_data = articles_df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Analysis CSV",
        data=csv_data,
        file_name="gold_market_sentiment.csv",
        mime="text/csv"
    )


else:

    # =====================================================
    # INITIAL SCREEN
    # =====================================================

    st.info(
        "👈 Select the number of articles and "
        "click **Run Analysis** to start."
    )


    st.markdown(
        """
        ### How this system works

        **1. News Collection**

        Google News RSS is used to collect recent
        financial news related to gold.

        **2. Data Processing**

        Duplicate articles are removed.

        **3. Sentiment Analysis**

        Two sentiment models analyze each headline:

        - **FinBERT** — specialized for financial text
        - **VADER** — general-purpose sentiment analysis

        **4. Model Comparison**

        The predictions of both models are compared.

        **5. Market Sentiment**

        FinBERT predictions are aggregated into an
        overall sentiment score.

        **6. Visualization**

        Results are displayed using charts and tables.
        """
    )