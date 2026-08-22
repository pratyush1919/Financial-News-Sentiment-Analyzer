# 📈 Gold Market Sentiment Analyzer

An NLP-based financial news sentiment analysis system that collects gold-related financial news and analyzes market sentiment using **FinBERT** and **VADER**.

The system compares the predictions of both sentiment models and presents the results through an interactive **Streamlit dashboard**.

---

## 🚀 Project Overview

Financial news can strongly influence investor perception of a market.

This project analyzes recent gold-related financial news and determines whether the overall news sentiment is:

- 🟢 Positive
- 🔴 Negative
- ⚪ Neutral

The system uses two different sentiment analysis approaches:

- **FinBERT** — a transformer-based model specifically trained for financial text.
- **VADER** — a lexicon and rule-based sentiment analysis model.

The predictions from both models are compared to understand where the models agree or disagree.

---

## ✨ Features

- 📰 Collects gold-related financial news using Google News RSS
- 🔎 Supports multiple financial search queries
- ♻️ Removes duplicate articles
- 🤖 Performs sentiment analysis using FinBERT
- 📊 Performs sentiment analysis using VADER
- 🔬 Compares predictions from both models
- 📈 Calculates an overall FinBERT sentiment score
- 🐂 Classifies market sentiment as Bullish, Neutral, or Bearish
- 📊 Interactive Streamlit dashboard
- 📋 Displays analyzed financial news in a table
- 💾 Allows analysis results to be downloaded as CSV
- 🗃️ Uses a local news cache when live news retrieval fails

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FinBERT | Financial sentiment analysis |
| VADER | Rule-based sentiment analysis |
| Hugging Face Transformers | Loading and running FinBERT |
| PyTorch | Deep learning framework used by FinBERT |
| Feedparser | Parsing Google News RSS feeds |
| Requests | Fetching web pages |
| BeautifulSoup | Extracting article text |
| Pandas | Data processing |
| Plotly | Interactive visualizations |
| Streamlit | Web dashboard |

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │   Google News RSS    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    News Collection   │
                 │     Feedparser       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Duplicate Removal    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   News Headlines     │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴───────────┐
                 ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐
        │     FinBERT     │    │      VADER      │
        │ Financial NLP   │    │ Rule-based NLP  │
        └────────┬────────┘    └────────┬────────┘
                 │                      │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Model Comparison     │
                 │ Agreement Analysis   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Overall Sentiment    │
                 │ Score & Classification│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Streamlit Dashboard  │
                 └──────────────────────┘