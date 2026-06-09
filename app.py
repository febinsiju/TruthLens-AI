from flask import Flask, render_template, request
import requests
import joblib

app = Flask(__name__)

API_KEY = "f1adadd6c1a24e43a95ab82fe7daa003"

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def analyze_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    try:
        score = model.decision_function(vector)[0]
        confidence = round(abs(score) / (abs(score) + 1) * 100, 2)
    except Exception:
        confidence = 85.0

    if confidence < 55:
        prediction = "REAL"

    explanation = (
        "This headline appears to follow reliable news language patterns."
        if prediction == "REAL"
        else "Suspicious or sensational language patterns detected."
    )

    return prediction, confidence, explanation


def fetch_news(query="latest"):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=publishedAt&pageSize=100&apiKey={API_KEY}"
    )

    response = requests.get(url, timeout=10)
    data = response.json()
    print(data)

    if data.get("status") == "ok":
        return data.get("articles", [])

    return []


def process_articles(articles):
    news_results = []

    for article in articles:
        title = article.get("title") or "No title"
        description = article.get("description") or ""
        image = article.get("urlToImage")
        source = article.get("source", {}).get("name", "Unknown Source")
        url = article.get("url", "#")

        prediction, confidence, explanation = analyze_news(title + " " + description)

        news_results.append({
            "title": title,
            "description": description,
            "image": image,
            "source": source,
            "url": url,
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation
        })

    return news_results


def render_page(news_results, page_title):
    real_count = sum(1 for news in news_results if news["prediction"] == "REAL")
    fake_count = sum(1 for news in news_results if news["prediction"] == "FAKE")

    return render_template(
        "index.html",
        news_results=news_results,
        real_count=real_count,
        fake_count=fake_count,
        page_title=page_title
    )


@app.route("/")
def home():
    articles = fetch_news("latest")
    news_results = process_articles(articles)
    return render_page(news_results, "Latest News")


@app.route("/search")
def search():
    query = request.args.get("query", "latest")
    articles = fetch_news(query)
    news_results = process_articles(articles)
    return render_page(news_results, f"Search Results for {query}")


@app.route("/real")
def real_news():
    articles = fetch_news("latest")
    news_results = process_articles(articles)
    real_only = [news for news in news_results if news["prediction"] == "REAL"]
    return render_page(real_only, "Verified Real News")


@app.route("/fake")
def fake_news():
    articles = fetch_news("latest")
    news_results = process_articles(articles)
    fake_only = [news for news in news_results if news["prediction"] == "FAKE"]
    return render_page(fake_only, "Suspicious / Fake News")


@app.route("/category/<topic>")
def category(topic):
    articles = fetch_news(topic)
    news_results = process_articles(articles)
    return render_page(news_results, topic.capitalize() + " News")


if __name__ == "__main__":
    app.run(debug=True)