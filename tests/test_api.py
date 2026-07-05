from fastapi.testclient import TestClient


def test_health_and_prediction(monkeypatch, test_model_path):
    import app.main as main

    main.MODEL = None
    monkeypatch.setattr(main, "MODEL_PATH", test_model_path)
    client = TestClient(main.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["risk"] in ["low", "high"]
    assert 0 <= body["confidence"] <= 1


def test_invalid_payload_returns_422(monkeypatch, test_model_path):
    import app.main as main

    main.MODEL = None
    monkeypatch.setattr(main, "MODEL_PATH", test_model_path)
    client = TestClient(main.app)
    response = client.post("/predict", json={"age": -5})
    assert response.status_code == 422
