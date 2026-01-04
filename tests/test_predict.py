# tests/test_predict.py
def test_predict_success(client, monkeypatch):

    class DummyModel:
        def predict(self, X):
            return [2]  # positive

    # Inject fake model
    monkeypatch.setattr("app.deps.model", DummyModel(), raising=False)

    response = client.post(
        "/predict",
        json={"text": "this is a good video"}
    )

    assert response.status_code == 200
    assert "sentiment_summary" in response.json
