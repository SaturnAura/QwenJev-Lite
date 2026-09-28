"""A thin HTTP surface shaped like the documented Jev API.

    POST /v1/decide
    {
      "state": "My payouts have failed three times. ...",
      "questions": {
        "queue": {"type": "choice", "instructions": "...", "criteria": {...}},
        "escalate": {"type": "bool", "instructions": "..."}
      }
    }

The response carries one distribution per question, the adapter's confidence field,
the billing figure and the timing split, so the essay's observations can be checked
against a running service rather than a notebook.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import QwenJevConfig
from .engine import QwenJevLite


class DecideRequest(BaseModel):
    state: str = Field(..., description="Shared state, encoded once per request")
    questions: dict[str, dict[str, Any]] = Field(..., description="Typed questions")
    share_state: bool | None = Field(
        default=None, description="Override: share one state encoding across branches"
    )


class DecideResponse(BaseModel):
    results: dict[str, Any]
    usage: dict[str, int]
    timing: dict[str, Any]


def create_app(
    engine: QwenJevLite | None = None,
    *,
    config: QwenJevConfig | None = None,
    fake: bool = False,
) -> FastAPI:
    app = FastAPI(title="QwenJev-lite", version="0.1.0")
    state: dict[str, Any] = {"engine": engine}
    lock = threading.Lock()

    def get_engine() -> QwenJevLite:
        if state["engine"] is None:
            with lock:
                if state["engine"] is None:
                    if fake:
                        from .testing import build_tiny_engine

                        state["engine"] = build_tiny_engine()
                    else:
                        state["engine"] = QwenJevLite.from_pretrained(config=config)
        return state["engine"]

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        loaded = state["engine"] is not None
        payload: dict[str, Any] = {"status": "ok", "model_loaded": loaded}
        if loaded:
            payload["readout"] = state["engine"].readout.name
            payload["cached_states"] = len(state["engine"]._state_cache)
        return payload

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        cfg = config or QwenJevConfig()
        return {
            "object": "list",
            "data": [
                {
                    "id": "qwenjev-lite",
                    "backbone": cfg.model_path,
                    "readout": cfg.readout,
                    "max_options": cfg.limits.max_options,
                    "max_branch_tokens": cfg.limits.max_branch_tokens,
                    "max_request_tokens": cfg.limits.max_request_tokens,
                }
            ],
        }

    @app.post("/v1/decide", response_model=DecideResponse)
    def decide(request: DecideRequest) -> DecideResponse:
        engine_ = get_engine()
        try:
            response = engine_.decide(
                request.state, request.questions, share_state=request.share_state
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DecideResponse(**response.to_dict())

    return app


def serve(*, host: str = "127.0.0.1", port: int = 8300, **kwargs) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(**kwargs), host=host, port=port)
