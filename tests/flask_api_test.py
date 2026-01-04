import pytest
from flask_app.app import app as flask_app
@pytest.fixture
def client(monkeypatch):
    flask_app.app.config["TESTING"] = True  

    # ---------- MOCK MODEL ----------
    class DummyModel:
        def predict(self, X):
            return [2]  # positive

    monkeypatch.setattr(flask_app, "model", DummyModel())

    # ---------- MOCK GROQ ----------
    class DummyGroq:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    class R:
                        choices = [
                            type("obj", (), {
                                "message": type("obj", (), {"content": "dummy answer"})
                            })
                        ]
                    return R()

    monkeypatch.setattr(flask_app, "client", DummyGroq())

    with flask_app.app.test_client() as client:
        yield client

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json["status"] == "ok"

def test_predict_single_text(client):
    res = client.post(
        "/predict",
        json={"text": "this is a good video"}
    )

    assert res.status_code == 200
    assert "sentiment_summary" in res.json

def test_ask_video(client):
    res = client.post(
        "/ask-video",
        json={
            "question": "summary",
            "video_context": "this video explains machine learning basics"
        }
    )

    assert res.status_code == 200
    assert "answer" in res.json
