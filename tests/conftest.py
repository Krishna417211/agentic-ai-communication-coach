"""Shared fixtures."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMError, LLMProvider
from app.main import create_app


class FakeLLM(LLMProvider):
    """Scripted provider so the LLM code paths are testable without a key.

    Responses are keyed by a substring of the prompt; the first match wins.
    """

    name = "fake"
    supports_generation = True

    def __init__(self, responses: dict[str, dict] | None = None, *, fail: bool = False):
        self.responses = responses or {}
        self.fail = fail
        self.calls: list[str] = []

    async def complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.calls.append(prompt)
        if self.fail:
            raise LLMError("simulated provider outage")
        for needle, payload in self.responses.items():
            if needle in prompt:
                return json.dumps(payload)
        return json.dumps({})


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_email() -> str:
    return (
        "hey john, i just wanted to quickly check if maybe you might have had a "
        "chance to possibly look at the report i sent? its been a while and the "
        "deadline was moved. thanks"
    )


@pytest.fixture
def hostile_message() -> str:
    return (
        "You never send the numbers on time. This is unacceptable and it's "
        "obviously your fault. I need them ASAP."
    )
