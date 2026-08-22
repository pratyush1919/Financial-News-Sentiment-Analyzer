# News Sentiment Scanner

A Python application that fetches news articles from Google News RSS feeds and analyzes their sentiment using natural language processing. This project is specifically configured to analyze gold market news, but can be easily adapted for other topics.

## 📋 Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Logic Explanation](#logic-explanation)

## ✨ Features

- **Automated News Fetching**: Retrieves news articles from Google News RSS feeds based on customizable search queries
- **Sentiment Analysis**: Analyzes article sentiment using VADER Sentiment Analyzer
- **Multiple Query Support**: Processes multiple search queries simultaneously
- **Sentiment Summary**: Provides a comprehensive summary of sentiment distribution across all analyzed articles
- **Article Content Extraction**: Fetches and extracts full article content from news websites

## 🔧 How It Works

The application follows this workflow:

1. **Query Execution**: Takes multiple search queries (e.g., "gold market", "gold price") and searches Google News RSS feeds
2. **Article Fetching**: Retrieves article titles, links, publication dates, and content
3. **Content Extraction**: Scrapes article content from the original news websites using BeautifulSoup
4. **Sentiment Analysis**: Analyzes each article's sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)
5. **Summary Generation**: Aggregates sentiment results and displays statistics

## 📦 Dependencies

### System Requirements
- **Python** (3.8+): Python programming language (tested with Python 3.12.10)
  - Python 3.8 or higher is required
  - Download from [python.org](https://www.python.org/downloads/)

### Core Libraries
- **feedparser** (6.0.11): Parses RSS feeds from Google News
- **requests** (2.32.3): HTTP library for fetching web content
- **beautifulsoup4** (4.13.4): HTML parsing and content extraction
- **vaderSentiment** (3.3.2): Sentiment analysis using VADER algorithm
- **textblob** (0.19.0): Text processing library (available but not currently used)

### Machine Learning (Optional)
- **transformers** (4.51.3): Hugging Face transformers library
- **torch** (2.7.0): PyTorch deep learning framework
- **numpy** (2.2.5): Numerical computing library

### Supporting Libraries
- **nltk** (3.9.1): Natural Language Toolkit
- **tokenizers** (0.21.1): Fast tokenization library
- **huggingface-hub** (0.30.2): Hugging Face model hub integration
- **tqdm** (4.67.1): Progress bar library
- **certifi**, **urllib3**, **charset-normalizer**: HTTP and SSL support

See `requirements.txt` for the complete list with specific versions.

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher** must be installed on your system
- Check your Python version: `python --version` or `python3 --version`
- If Python is not installed, download it from [python.org](https://www.python.org/downloads/)

### Quick Setup (Windows)

1. **Run the setup script**:
   ```batch
   run.bat
   ```
   This script will:
   - Check if a virtual environment exists, create one if it doesn't
   - Activate the virtual environment
   - Install all required dependencies

### Manual Setup

1. **Create a virtual environment**:
   ```batch
   python -m venv .venv
   ```

2. **Activate the virtual environment**:
   ```batch
   .venv\Scripts\activate.bat
   ```

3. **Install dependencies**:
   ```batch
   pip install --default-timeout=1000 -r requirements.txt
   ```
   Note: The `--default-timeout=1000` flag helps with downloading large packages like PyTorch.

### Linux/Mac Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install --default-timeout=1000 -r requirements.txt
```

## 💻 Usage

### Basic Usage

1. **Activate the virtual environment** (if not already activated):
   ```batch
   .venv\Scripts\activate.bat
   ```

2. **Run the main script**:
   ```batch
   python sentiment_analysis.py
   ```

### Customizing Search Queries

Edit the `queries` list in the `main()` function of `sentiment_analysis.py`:

```python
queries = [
    "gold market",
    "gold price",
    "your custom query here"
]
```

### Adjusting Number of Articles

Modify the `num_articles_per_query` variable:

```python
num_articles_per_query = 10  # Change this number
```

## 📁 Project Structure

```
NewsSentimentScanner/
│
├── sentiment_analysis.py    # Main application script
├── test.ipynb               # Jupyter notebook for testing
├── requirements.txt         # Python dependencies
├── run.bat                  # Windows setup script
├── README.md                # This file
└── .venv/                   # Virtual environment (created during setup)
```

## 🧠 Logic Explanation

### 1. News Fetching (`fetch_news`)

```python
def fetch_news(query, num_articles=10):
```

- Constructs a Google News RSS URL with the search query (URL-encoded)
- Parses the RSS feed using `feedparser`
- Extracts article metadata (title, link, publication date)
- Fetches full article content for each article
- Returns a list of article dictionaries

**Key Logic**:
- Uses Google News RSS API: `https://news.google.com/rss/search?q={query}`
- Limits results to `num_articles` per query
- Each article includes title, link, published date, and scraped content

### 2. Content Extraction (`fetch_article_content`)

```python
def fetch_article_content(url):
```

- Makes HTTP GET request to the article URL
- Parses HTML using BeautifulSoup
- Extracts all paragraph (`<p>`) tags
- Combines paragraphs into a single text string
- Handles errors gracefully (returns "Content not retrieved." on failure)

**Key Logic**:
- Uses `requests` for HTTP requests with 10-second timeout
- BeautifulSoup extracts text from HTML paragraphs
- Error handling prevents crashes from inaccessible websites

### 3. Sentiment Analysis (`analyze_sentiment`)

```python
def analyze_sentiment(text):
```

- Uses VADER Sentiment Analyzer to compute sentiment scores
- Calculates compound polarity score (ranges from -1 to +1)
- Classifies sentiment based on thresholds:
  - **Positive**: polarity > 0.05
  - **Negative**: polarity < -0.05
  - **Neutral**: -0.05 ≤ polarity ≤ 0.05

**Key Logic**:
- VADER is specifically designed for social media text and news
- Compound score aggregates positive, negative, and neutral scores
- Thresholds can be adjusted for different sensitivity levels

**Note**: The code includes commented-out FinBERT implementation (financial sentiment model) that can be enabled for more domain-specific analysis.

### 4. Sentiment Summarization (`summarize_sentiments`)

```python
def summarize_sentiments(articles):
```

- Iterates through all analyzed articles
- Counts occurrences of each sentiment category
- Calculates percentage distribution
- Displays formatted summary statistics

**Key Logic**:
- Aggregates individual article sentiments
- Provides overall market sentiment overview
- Shows both count and percentage for each sentiment

### 5. Main Execution Flow (`main`)

```python
def main():
```

1. Defines multiple search queries related to gold market
2. Fetches articles for each query
3. Combines all articles into a single list
4. Analyzes sentiment for each article
5. Displays individual article results
6. Generates and displays summary statistics

**Key Logic**:
- Processes multiple queries to get comprehensive coverage
- Currently analyzes article titles (can be changed to analyze full content)
- Provides both detailed and summary views

## 🔄 Alternative Sentiment Models

The code includes a commented-out FinBERT implementation. To use it:

1. Uncomment the FinBERT code in `analyze_sentiment`
2. Comment out the VADER implementation
3. FinBERT provides financial domain-specific sentiment analysis

FinBERT uses:
- Pre-trained model: `yiyanghkust/finbert-tone`
- Three labels: Positive, Negative, Neutral
- Confidence scores for each prediction

## ⚠️ Notes

- **Internet Connection Required**: The application fetches live news from Google News RSS feeds
- **Rate Limiting**: Be mindful of making too many requests in a short time
- **Content Availability**: Some articles may not be accessible due to paywalls or access restrictions
- **Processing Time**: Large numbers of articles may take time to process, especially if using FinBERT

## 🐛 Troubleshooting

### Installation Issues

- **Timeout Errors**: Use `--default-timeout=1000` flag when installing
- **PyTorch Installation**: Ensure you have sufficient disk space (~2GB for PyTorch)
- **SSL Errors**: Update certificates: `pip install --upgrade certifi`

### Runtime Issues

- **No Articles Found**: Check internet connection and query terms
- **Content Extraction Fails**: Some websites block scraping; this is expected
- **Import Errors**: Ensure virtual environment is activated

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to modify the queries, add new features, or improve the sentiment analysis methods!

