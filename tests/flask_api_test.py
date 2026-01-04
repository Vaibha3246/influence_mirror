import os
import pytest
import requests

# -------------------------
# CI CONFIG
# -------------------------
BASE_URL = " http://127.0.0.1:8000"


# -------------------------
# HEALTH CHECK
# -------------------------
def test_health_endpoint():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# -------------------------
# PREDICT (MODEL DISABLED)
# -------------------------
def test_predict_model_not_loaded():
    payload = {
        "texts": ["This is a test comment"],
        "category": "education"
    }

    r = requests.post(f"{BASE_URL}/predict", json=payload)

    # In CI → model is disabled
    assert r.status_code == 503
    assert "Model not loaded" in r.text


# -------------------------
# ASK VIDEO (LLM DISABLED)
# -------------------------
def test_ask_video_disabled_in_test_mode():
    payload = {
        "question": "Give summary",
        "video_context": "This video explains machine learning basics."
    }

    r = requests.post(f"{BASE_URL}/ask-video", json=payload)

    assert r.status_code == 503
    assert "LLM disabled" in r.text


# -------------------------
# VIDEO TOPICS (SAFE)
# -------------------------
def test_video_topics_endpoint():
    payload = {
        "video_context": "This video explains Python basics, loops, and functions."
    }

    r = requests.post(f"{BASE_URL}/video-topics", json=payload)

    assert r.status_code in [200, 503]
    # 503 allowed if LLM blocked

def test_suggested_questions_endpoint():
    payload = {
        "video_context": "This video explains data science and machine learning.",
        "chat_history": [
            {"role": "user", "content": "What is data science?"},
            {"role": "assistant", "content": "Data science is about data."}
        ]
    }

    r = requests.post(f"{BASE_URL}/suggested-questions", json=payload)

    assert r.status_code in [200, 503]

# -------------------------
# INVALID INPUT HANDLING
# -------------------------
def test_predict_invalid_payload():
    r = requests.post(f"{BASE_URL}/predict", json={})
    assert r.status_code in [400, 503]
