from fastapi.testclient import TestClient

from qwenjev.api import create_app
from qwenjev.testing import build_tiny_engine

QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"payments": "Payouts", "account": "Login", "other": "Else"},
    },
    "escalate": {"type": "bool", "instructions": "Urgent?"},
}


def client() -> TestClient:
    return TestClient(create_app(engine=build_tiny_engine()))


def test_healthz():
    payload = client().get("/healthz").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_models_lists_the_limits():
    payload = client().get("/v1/models").json()
    model = payload["data"][0]
    assert model["max_options"] == 255
    assert model["max_branch_tokens"] == 32768
    assert model["max_request_tokens"] == 65536


def test_decide_returns_parallel_distributions():
    response = client().post(
        "/v1/decide", json={"state": "Payouts failed three times.", "questions": QUESTIONS}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["results"]) == {"queue", "escalate"}
    assert abs(sum(body["results"]["queue"]["probabilities"].values()) - 1.0) < 1e-5
    assert body["usage"]["output_tokens"] == 4 + 2 * 15 + 5 + 8


def test_invalid_question_returns_400():
    response = client().post(
        "/v1/decide",
        json={"state": "x", "questions": {"q": {"type": "choice", "instructions": "?"}}},
    )
    assert response.status_code == 400
    assert "criteria" in response.json()["detail"]
