import os
import pytest
from flask_app.app import app   # apna correct path rakho

# -------------------------
# TEST CLIENT (VERY IMPORTANT)
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
# PREDICT (MODEL DISABLED)
# -------------------------
def test_predict_model_not_loaded(client):
    payload = {
        "texts": ["This is a test comment"],
        "category": "education"
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 503
    assert "Model not loaded" in r.json["error"]


# -------------------------
# ASK VIDEO (LLM DISABLED)
# -------------------------
def test_ask_video_disabled_in_test_mode(client):
    payload = {
        "question": "Give summary",
        "video_context": "This video explains machine learning basics."
    }

    r = client.post("/ask-video", json=payload)
    assert r.status_code == 503
    assert "LLM disabled" in r.json["error"]


# -------------------------
# VIDEO TOPICS
# -------------------------
def test_video_topics_endpoint(client):
    payload = {
        "video_context": "Python basics, loops, and functions."
    }

    r = client.post("/video-topics", json=payload)
    assert r.status_code in [200, 503]


def test_suggested_questions_endpoint(client):
    payload = {
        "video_context": "Data science and machine learning",
        "chat_history": [
            {"role": "user", "content": "What is data science?"}
        ]
    }

    r = client.post("/suggested-questions", json=payload)
    assert r.status_code in [200, 503]


# -------------------------
# INVALID INPUT
# -------------------------
def test_predict_invalid_payload(client):
    r = client.post("/predict", json={})
    assert r.status_code in [400, 503]
