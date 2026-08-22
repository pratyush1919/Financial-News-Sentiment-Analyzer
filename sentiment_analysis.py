import json
import os
import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification


# =========================================================
# FINBERT SETUP
# =========================================================

FINBERT_MODEL = "yiyanghkust/finbert-tone"

print("Loading FinBERT model...")

finbert_tokenizer = BertTokenizer.from_pretrained(
    FINBERT_MODEL
)

finbert_model = BertForSequenceClassification.from_pretrained(
    FINBERT_MODEL
)

# Prediction mode
finbert_model.eval()

# Correct label order for yiyanghkust/finbert-tone
finbert_labels = [
    "Neutral",
    "Positive",
    "Negative"
]

print("FinBERT loaded successfully.")


# =========================================================
# LOCAL CACHE
# =========================================================

CACHE_FILE = "news_cache.json"


def save_articles_to_cache(articles):
    """
    Save news articles locally so they can be reused
    if live news fetching fails later.
    """

    try:
        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                articles,
                file,
                ensure_ascii=False,
                indent=4
            )

        print(
            f"\nNews articles saved to {CACHE_FILE}"
        )

    except OSError as e:

        print(
            f"Could not save news cache: {e}"
        )


def load_articles_from_cache():
    """
    Load previously saved articles from local cache.
    """

    if not os.path.exists(CACHE_FILE):
        return []

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            articles = json.load(file)

        print(
            f"\nLoaded {len(articles)} "
            f"articles from local cache."
        )

        return articles

    except (
        json.JSONDecodeError,
        OSError
    ):

        print(
            "Could not read the local news cache."
        )

        return []


# =========================================================
# NEWS COLLECTION
# =========================================================


def fetch_news(query, num_articles=10):
    """
    Fetch financial news from Google News RSS.
    """

    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={quote(query)}"
    )

    feed = feedparser.parse(rss_url)

    news_items = feed.entries[:num_articles]

    articles = []

    for item in news_items:

        title = item.get(
            "title",
            "Unknown"
        )

        link = item.get(
            "link",
            ""
        )

        published = item.get(
            "published",
            "Unknown"
        )

        content = fetch_article_content(
            link
        )

        articles.append(
            {
                "title": title,
                "link": link,
                "published": published,
                "content": content
            }
        )

    return articles


def fetch_article_content(url):
    """
    Extract readable text from an article webpage.
    """

    if not url:
        return "Content not retrieved."

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        content = " ".join(
            paragraph.get_text(
                " ",
                strip=True
            )
            for paragraph in paragraphs
        )

        if not content.strip():
            return "Content not retrieved."

        return content.strip()

    except requests.RequestException:

        return "Content not retrieved."


# =========================================================
# FINBERT SENTIMENT ANALYSIS
# =========================================================


def analyze_sentiment_finbert(text):
    """
    Analyze financial sentiment using FinBERT.

    Returns:
        confidence
        sentiment
    """

    if not text or not text.strip():
        return 0.0, "Neutral"

    try:

        inputs = finbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = finbert_model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        ).numpy()[0]

        max_index = np.argmax(
            probabilities
        )

        sentiment = finbert_labels[
            max_index
        ]

        confidence = float(
            probabilities[max_index]
        )

        return confidence, sentiment

    except Exception as e:

        print(
            f"FinBERT analysis failed: {e}"
        )

        return 0.0, "Neutral"


# =========================================================
# VADER SENTIMENT ANALYSIS
# =========================================================


vader_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER.

    Returns:
        polarity
        sentiment
    """

    if not text or not text.strip():
        return 0.0, "Neutral"

    try:

        scores = vader_analyzer.polarity_scores(
            text
        )

        polarity = scores["compound"]

        if polarity > 0.05:

            sentiment = "Positive"

        elif polarity < -0.05:

            sentiment = "Negative"

        else:

            sentiment = "Neutral"

        return polarity, sentiment

    except Exception as e:

        print(
            f"VADER analysis failed: {e}"
        )

        return 0.0, "Neutral"


# =========================================================
# ARTICLE ANALYSIS
# =========================================================


def analyze_article(article):
    """
    Analyze one financial news article using
    both FinBERT and VADER.

    The headline and article content are combined.
    """

    title = article.get(
        "title",
        ""
    )

    content = article.get(
        "content",
        ""
    )

    # -----------------------------------------------------
    # COMBINE TITLE + CONTENT
    # -----------------------------------------------------

    if (
        content
        and content.strip()
        and content != "Content not retrieved."
    ):

        text = (
            f"{title}. "
            f"{content}"
        )

    else:

        # Fallback to headline
        text = title

    # Limit extremely large webpages.
    # FinBERT itself is limited to 512 tokens.
    text = text[:5000]

    # -----------------------------------------------------
    # FINBERT
    # -----------------------------------------------------

    (
        finbert_confidence,
        finbert_sentiment
    ) = analyze_sentiment_finbert(
        text
    )

    # -----------------------------------------------------
    # VADER
    # -----------------------------------------------------

    (
        vader_polarity,
        vader_sentiment
    ) = analyze_sentiment_vader(
        text
    )

    # -----------------------------------------------------
    # MODEL AGREEMENT
    # -----------------------------------------------------

    agreement = (
        finbert_sentiment
        == vader_sentiment
    )

    return {
        "finbert_sentiment":
            finbert_sentiment,

        "finbert_confidence":
            finbert_confidence,

        "vader_sentiment":
            vader_sentiment,

        "vader_polarity":
            vader_polarity,

        "agreement":
            agreement
    }


# =========================================================
# SUMMARY
# =========================================================


def summarize_sentiments(results):

    finbert_summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    vader_summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    agreement_count = 0

    # -----------------------------------------------------
    # PROCESS RESULTS
    # -----------------------------------------------------

    for result in results:

        finbert_sentiment = (
            result["finbert_sentiment"]
        )

        vader_sentiment = (
            result["vader_sentiment"]
        )

        finbert_summary[
            finbert_sentiment
        ] += 1

        vader_summary[
            vader_sentiment
        ] += 1

        if result["agreement"]:
            agreement_count += 1

    total = len(results)

    # -----------------------------------------------------
    # FINBERT PERCENTAGES
    # -----------------------------------------------------

    if total > 0:

        positive_percentage = (
            finbert_summary["Positive"]
            / total
        ) * 100

        negative_percentage = (
            finbert_summary["Negative"]
            / total
        ) * 100

        neutral_percentage = (
            finbert_summary["Neutral"]
            / total
        ) * 100

    else:

        positive_percentage = 0
        negative_percentage = 0
        neutral_percentage = 0

    # -----------------------------------------------------
    # OVERALL MARKET SENTIMENT SCORE
    # -----------------------------------------------------
    #
    # Positive = +1
    # Neutral  =  0
    # Negative = -1
    #
    # Score range:
    #
    # +1 = Extremely Positive
    #  0 = Neutral
    # -1 = Extremely Negative
    #
    # FinBERT is used as the primary model because
    # it is specifically designed for financial text.
    # -----------------------------------------------------

    sentiment_score = (
        finbert_summary["Positive"]
        - finbert_summary["Negative"]
    ) / total if total > 0 else 0

    # -----------------------------------------------------
    # OVERALL SENTIMENT
    # -----------------------------------------------------

    if sentiment_score > 0.20:

        overall_sentiment = "Positive"

    elif sentiment_score < -0.20:

        overall_sentiment = "Negative"

    else:

        overall_sentiment = "Neutral"

    # -----------------------------------------------------
    # MARKET CONDITION
    # -----------------------------------------------------

    if sentiment_score > 0.50:

        market_condition = "Strongly Bullish"

    elif sentiment_score > 0.20:

        market_condition = "Bullish"

    elif sentiment_score < -0.50:

        market_condition = "Strongly Bearish"

    elif sentiment_score < -0.20:

        market_condition = "Bearish"

    else:

        market_condition = "Neutral / Mixed"

    # -----------------------------------------------------
    # MODEL AGREEMENT
    # -----------------------------------------------------

    agreement_percentage = (
        agreement_count / total * 100
        if total > 0
        else 0
    )

    disagreement_count = (
        total - agreement_count
    )

    disagreement_percentage = (
        disagreement_count / total * 100
        if total > 0
        else 0
    )

    # =====================================================
    # FINANCIAL NEWS SUMMARY
    # =====================================================

    print("\n")
    print("=" * 60)
    print("FINANCIAL NEWS SENTIMENT SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal articles analyzed: {total}"
    )

    # =====================================================
    # FINBERT
    # =====================================================

    print("\n--- FinBERT ---")

    print(
        f"Positive: "
        f"{finbert_summary['Positive']} "
        f"({positive_percentage:.2f}%)"
    )

    print(
        f"Negative: "
        f"{finbert_summary['Negative']} "
        f"({negative_percentage:.2f}%)"
    )

    print(
        f"Neutral:  "
        f"{finbert_summary['Neutral']} "
        f"({neutral_percentage:.2f}%)"
    )

    # =====================================================
    # VADER
    # =====================================================

    print("\n--- VADER ---")

    for sentiment, count in (
        vader_summary.items()
    ):

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{sentiment}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    print("\n--- Model Comparison ---")

    print(
        f"Agreement: "
        f"{agreement_count} "
        f"({agreement_percentage:.2f}%)"
    )

    print(
        f"Disagreement: "
        f"{disagreement_count} "
        f"({disagreement_percentage:.2f}%)"
    )

    # =====================================================
    # OVERALL MARKET SENTIMENT
    # =====================================================

    print("\n")
    print("=" * 60)
    print("OVERALL GOLD MARKET SENTIMENT")
    print("=" * 60)

    print(
        f"\nFinBERT Sentiment Score: "
        f"{sentiment_score:+.2f}"
    )

    print(
        f"Overall Sentiment: "
        f"{overall_sentiment}"
    )

    print(
        f"Market Condition: "
        f"{market_condition}"
    )

    print(
        "\nScore Interpretation:"
    )

    print(
        "  +1.00 = Extremely Positive"
    )

    print(
        "   0.00 = Neutral"
    )

    print(
        "  -1.00 = Extremely Negative"
    )

    print("=" * 60)


# =========================================================
# MAIN
# =========================================================


def main():

    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment"
    ]

    num_articles_per_query = 10

    all_articles = []

    # =====================================================
    # FETCH NEWS
    # =====================================================

    print("\n" + "=" * 60)
    print("FETCHING FINANCIAL NEWS")
    print("=" * 60)

    try:

        for query in queries:

            print(
                f"\nFetching news articles "
                f"for '{query}'..."
            )

            try:

                articles = fetch_news(
                    query,
                    num_articles_per_query
                )

                print(
                    f"Retrieved {len(articles)} "
                    f"articles."
                )

                all_articles.extend(
                    articles
                )

            except Exception as e:

                print(
                    f"Failed to fetch "
                    f"'{query}': {e}"
                )

    except Exception as e:

        print(
            f"\nLive news fetching failed: {e}"
        )

    # =====================================================
    # FALLBACK TO CACHE
    # =====================================================

    if not all_articles:

        print(
            "\nNo live articles were retrieved."
        )

        print(
            "Trying local news cache..."
        )

        all_articles = (
            load_articles_from_cache()
        )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    unique_articles = {}

    for article in all_articles:

        link = article.get(
            "link",
            ""
        )

        if link:

            unique_articles[
                link
            ] = article

    all_articles = list(
        unique_articles.values()
    )

    print(
        f"\nTotal unique articles: "
        f"{len(all_articles)}\n"
    )

    # =====================================================
    # SAVE CACHE
    # =====================================================

    if all_articles:

        save_articles_to_cache(
            all_articles
        )

    # =====================================================
    # NO ARTICLES
    # =====================================================

    if not all_articles:

        print(
            "\nNo articles available "
            "for sentiment analysis."
        )

        return

    # =====================================================
    # ANALYZE ARTICLES
    # =====================================================

    results = []

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING SENTIMENT ANALYSIS"
    )

    print(
        "=" * 60
    )

    for idx, article in enumerate(
        all_articles,
        1
    ):

        analysis = analyze_article(
            article
        )

        results.append(
            analysis
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"Article {idx}: "
            f"{article.get('title', 'Unknown')}"
        )

        print(
            f"Published: "
            f"{article.get('published', 'Unknown')}"
        )

        print(
            f"Link: "
            f"{article.get('link', 'Unknown')}"
        )

        # =================================================
        # FINBERT
        # =================================================

        print("\nFinBERT:")

        print(
            f"Sentiment: "
            f"{analysis['finbert_sentiment']}"
        )

        print(
            f"Confidence: "
            f"{analysis['finbert_confidence'] * 100:.2f}%"
        )

        # =================================================
        # VADER
        # =================================================

        print("\nVADER:")

        print(
            f"Sentiment: "
            f"{analysis['vader_sentiment']}"
        )

        print(
            f"Polarity: "
            f"{analysis['vader_polarity']:.2f}"
        )

        # =================================================
        # MODEL COMPARISON
        # =================================================

        print("\nModel Comparison:")

        if analysis["agreement"]:

            print(
                "Agreement: YES"
            )

        else:

            print(
                "Agreement: NO"
            )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    summarize_sentiments(
        results
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================


if __name__ == "__main__":

    main()