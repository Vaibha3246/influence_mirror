import os
import pytest
from flask_app.app import app


# -------------------------
# TEST CLIENT
# -------------------------
@pytest.fixture
def client():
    os.environ["IS_TESTING"] = "true"
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


# -------------------------
# HEALTH CHECK
# -------------------------
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok"


# -------------------------
# PREDICT (TEST MODE DUMMY RESPONSE)
# -------------------------
def test_predict_test_mode(client):
    payload = {
        "texts": ["This is a test comment"],
        "category": "education"
    }

    r = client.post("/predict", json=payload)

    assert r.status_code == 200
    assert "sentiment_summary" in r.json
    assert "predictions" in r.json


# -------------------------
# ASK VIDEO (LLM DISABLED)
# -------------------------
def test_ask_video_disabled_in_test_mode(client):
    payload = {
        "question": "Give summary",
        "video_context": "This video explains ML basics."
    }

    r = client.post("/ask-video", json=payload)

    assert r.status_code == 503
    assert "LLM disabled" in r.json["error"]


# -------------------------
# VIDEO TOPICS 
# -------------------------
def test_video_topics_endpoint(client):
    payload = {
        "video_context": "Python basics, loops, functions."
    }

    r = client.post("/video-topics", json=payload)

    assert r.status_code == 200
    assert "topics" in r.json


# -------------------------
# SUGGESTED QUESTIONS (LLM DISABLED)
# -------------------------
def test_suggested_questions_disabled(client):
    payload = {
        "video_context": "Data science and ML",
        "chat_history": [
            {"role": "user", "content": "What is data science?"}
        ]
    }

    r = client.post("/suggested-questions", json=payload)

    assert r.status_code == 503
    assert "LLM disabled" in r.json["error"]


# -------------------------
# INVALID INPUT (PREDICT)
# -------------------------
def test_predict_invalid_payload(client):
    r = client.post("/predict", json={})

    # test-mode dummy still returns 200
    assert r.status_code == 200
