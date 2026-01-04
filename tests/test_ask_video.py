# tests/test_ask_video.py
def test_ask_video_success(client):
    response = client.post(
        "/ask-video",
        json={
            "question": "summarize",
            "video_context": "machine learning basics"
        }
    )

    assert response.status_code == 200
    assert "answer" in response.json


# tests/test_ask_video.py
def test_ask_video_missing_question(client):
    response = client.post(
        "/ask-video",
        json={"video_context": "ml"}
    )

    assert response.status_code == 400
