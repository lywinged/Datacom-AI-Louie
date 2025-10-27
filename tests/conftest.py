"""Shared fixtures for pytest-based integration and unit tests."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import AsyncGenerator, Iterable


import pytest
from fastapi.testclient import TestClient

# Ensure the backend package is importable when tests run from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INNER_BACKEND_PATH = PROJECT_ROOT / "backend" / "backend"
if "backend" not in sys.modules:
    backend_pkg = types.ModuleType("backend")
    backend_pkg.__path__ = [str(INNER_BACKEND_PATH)]
    sys.modules["backend"] = backend_pkg

# Provide a lightweight stub for onnxruntime to avoid heavy native dependencies during tests
ort_stub = types.ModuleType("onnxruntime")


def _stub_available_providers() -> list[str]:
    return ["CPUExecutionProvider"]


class _StubSessionOptions:
    def __init__(self):
        self.graph_optimization_level = 0
        self.intra_op_num_threads = 1
        self.inter_op_num_threads = 1


class _StubGraphOptimizationLevel:
    ORT_ENABLE_ALL = 0
    ORT_ENABLE_EXTENDED = 0


class _StubInferenceSession:
    def __init__(self, *args, **kwargs):
        self._outputs = [types.SimpleNamespace(shape=(1, 1, 1024))]

    def get_outputs(self):
        return self._outputs

    def run(self, *args, **kwargs):
        return [[[[0.0 for _ in range(1024)]]]]

    def get_providers(self):
        return ["CPUExecutionProvider"]


ort_stub.get_available_providers = _stub_available_providers
ort_stub.SessionOptions = _StubSessionOptions
ort_stub.GraphOptimizationLevel = _StubGraphOptimizationLevel
ort_stub.InferenceSession = _StubInferenceSession
sys.modules["onnxruntime"] = ort_stub

# Provide a lightweight tokenizer stub to bypass large model downloads during tests
tokenizer_module = types.ModuleType("transformers")


class _StubTokenizer:
    def __call__(self, inputs, **kwargs):
        batch_size = len(inputs)
        return {
            "input_ids": [[0] for _ in range(batch_size)],
            "attention_mask": [[1] for _ in range(batch_size)],
        }


class _StubAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _StubTokenizer()


tokenizer_module.AutoTokenizer = _StubAutoTokenizer
sys.modules["transformers"] = tokenizer_module

# Provide a lightweight qdrant_client stub to avoid loading native libraries
qdrant_models_module = types.ModuleType("qdrant_client.http.models")


class _StubVectorParams:
    def __init__(self, size: int, distance: str):
        self.size = size
        self.distance = distance


class _StubDistance:
    COSINE = "Cosine"


qdrant_models_module.VectorParams = _StubVectorParams
qdrant_models_module.Distance = _StubDistance
sys.modules["qdrant_client.http.models"] = qdrant_models_module

qdrant_exc_module = types.ModuleType("qdrant_client.http.exceptions")


class _StubUnexpectedResponse(Exception):
    def __init__(self, status_code: int | None = None):
        super().__init__("Unexpected response")
        self.status_code = status_code


qdrant_exc_module.UnexpectedResponse = _StubUnexpectedResponse
sys.modules["qdrant_client.http.exceptions"] = qdrant_exc_module

qdrant_http_module = types.ModuleType("qdrant_client.http")
qdrant_http_module.models = qdrant_models_module
qdrant_http_module.exceptions = qdrant_exc_module
sys.modules["qdrant_client.http"] = qdrant_http_module

qdrant_module = types.ModuleType("qdrant_client")


class _StubCollectionInfo:
    def __init__(self, vector_size: int):
        vectors = types.SimpleNamespace(size=vector_size)
        params = types.SimpleNamespace(vectors=vectors)
        self.config = types.SimpleNamespace(params=params)
        self.vectors_count = 0
        self.segments_count = 0
        self.status = "green"


class _StubQdrantClient:
    def __init__(self, host: str | None = None, port: int | None = None, path: str | None = None):
        self.host = host
        self.port = port
        self.path = path
        self._collections: dict[str, _StubCollectionInfo] = {}

    def get_collection(self, collection_name: str):
        try:
            return self._collections[collection_name]
        except KeyError as exc:
            raise _StubUnexpectedResponse(status_code=404) from exc

    def create_collection(self, collection_name: str, vectors_config: _StubVectorParams):
        self._collections[collection_name] = _StubCollectionInfo(vectors_config.size)

    def ensure_collection_ready(self, collection_name: str):
        return self.get_collection(collection_name)


qdrant_module.QdrantClient = _StubQdrantClient
qdrant_module.http = qdrant_http_module
sys.modules["qdrant_client"] = qdrant_module

# Stub for backend.services.qdrant_seed to avoid network requests during tests
seed_module = types.ModuleType("backend.services.qdrant_seed")
seed_module.ensure_seed_collection = lambda **kwargs: {"skipped": 0}
seed_module.get_seed_status = lambda **kwargs: {
    "status": "ready",
    "collection": "assessment_docs_minilm",
    "points_count": 0,
    "indexed": 0,
    "seed_file": None
}
sys.modules["backend.services.qdrant_seed"] = seed_module

# Stub for backend.services.onnx_inference to avoid heavy native dependencies and numpy
onnx_service_module = types.ModuleType("backend.services.onnx_inference")


class _StubEmbeddingModel:
    configured_path = "./models/stub-embed"
    resolved_model_path = "./models/stub-embed/model.onnx"
    vector_size = 1024

    def encode(self, texts, **kwargs):
        # Return simple unit vectors to avoid numpy dependency
        return [[0.0] * self.vector_size for _ in texts]


class _StubRerankerModel:
    configured_path = "./models/stub-reranker"
    resolved_model_path = "./models/stub-reranker/model.onnx"

    def score(self, query, documents, **kwargs):
        return [1.0 for _ in documents]

    def is_cpu_only(self):
        return True


_EMBED_MODEL = _StubEmbeddingModel()
_RERANK_MODEL = _StubRerankerModel()
_CURRENT_RERANK_PATH = _RERANK_MODEL.resolved_model_path


def _get_embedding_model():
    return _EMBED_MODEL


def _get_reranker_model():
    return _RERANK_MODEL


def _reranker_is_cpu_only():
    return True


def _switch_to_fallback_reranker():
    global _CURRENT_RERANK_PATH
    _CURRENT_RERANK_PATH = "./models/stub-reranker/model.onnx"


def _switch_to_fallback_mode():
    """Stub for switching to fallback mode (CPU inference)."""
    return True


def _switch_to_primary_mode():
    """Stub for switching to primary mode."""
    return True


def _set_reranker_model_path(path: str):
    global _CURRENT_RERANK_PATH
    _CURRENT_RERANK_PATH = path
    _RERANK_MODEL.resolved_model_path = path


def _get_current_reranker_path():
    return _CURRENT_RERANK_PATH


def _get_current_embed_path():
    return _EMBED_MODEL.resolved_model_path


def _has_cuda_available_stub():
    return False


onnx_service_module.get_embedding_model = _get_embedding_model
onnx_service_module.get_reranker_model = _get_reranker_model
onnx_service_module.reranker_is_cpu_only = _reranker_is_cpu_only
onnx_service_module.switch_to_fallback_reranker = _switch_to_fallback_reranker
onnx_service_module.switch_to_fallback_mode = _switch_to_fallback_mode
onnx_service_module.switch_to_primary_mode = _switch_to_primary_mode
onnx_service_module.set_reranker_model_path = _set_reranker_model_path
onnx_service_module.get_current_reranker_path = _get_current_reranker_path
onnx_service_module.get_current_embed_path = _get_current_embed_path
onnx_service_module._has_cuda_available = _has_cuda_available_stub

sys.modules["backend.services.onnx_inference"] = onnx_service_module

# ---------------------------------------------------------------------------
# Global environment overrides for predictable test behaviour
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _configure_test_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterable[None]:
    """Configure environment variables used by the services during tests."""
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("OPENAI_API_KEY", "unit-test-key")
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    os.environ.setdefault("ENABLE_REMOTE_INFERENCE", "false")

    metrics_dir = tmp_path_factory.mktemp("metrics")
    os.environ["PLANNING_METRICS_PATH"] = str(metrics_dir / "planning_metrics.json")

    yield


# ---------------------------------------------------------------------------
# Core FastAPI app fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app():
    """Return the FastAPI application instance."""
    from backend.main import app as fastapi_app

    return fastapi_app


@pytest.fixture(scope="session")
def client(app) -> Iterable[TestClient]:
    """Provide a shared TestClient for API integration tests."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Stub service fixtures used to isolate external dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_code_assistant(monkeypatch):
    """Patch the code assistant singleton with a lightweight stub."""
    from backend.models.code_schemas import (
        CodeResponse,
        Language,
        TestResult,
    )

    class _DummyAssistant:
        model_name = "dummy-code-assistant"

        async def generate_code(self, request) -> CodeResponse:  # pragma: no cover - exercised via API tests
            return CodeResponse(
                code="def add(a: int, b: int) -> int:\n    return a + b\n",
                language=request.language if isinstance(request.language, Language) else Language(request.language),
                test_passed=True,
                final_test_result=TestResult(
                    passed=True,
                    stdout="=== Program Output ===\n3\n",
                    stderr="",
                    exit_code=0,
                    execution_time_ms=12.5,
                    samples=[],
                ),
                retry_attempts=[],
                total_retries=0,
                generation_time_ms=145.0,
                tokens_used=42,
                cost_usd=0.01,
                token_usage={"prompt": 20, "completion": 22, "total": 42},
                token_cost_usd=0.01,
                initial_plan_summary="Stub plan summary",
                initial_plan_steps=["Collect requirements", "Generate implementation"],
                samples=None,
            )

    assistant = _DummyAssistant()
    monkeypatch.setattr("backend.routers.code_routes.get_code_assistant", lambda: assistant)
    return assistant


@pytest.fixture
def dummy_planning_agent(monkeypatch):
    """Patch the planning agent singleton with a deterministic stub."""
    from backend.models.agent_schemas import (
        PlanResponse,
        ReasoningStep,
        ToolCall,
        TripItinerary,
    )

    class _DummyPlanningAgent:
        model_name = "dummy-planning-agent"

        async def create_plan(self, request) -> PlanResponse:  # pragma: no cover - exercised via API tests
            destination = (
                request.constraints.destination_city
                if request.constraints and request.constraints.destination_city
                else "Test City"
            )
            currency = (
                request.constraints.currency
                if request.constraints and request.constraints.currency
                else "USD"
            )
            itinerary = TripItinerary(
                destination=destination,
                flights=[],
                weather_forecast=[],
                attractions=[],
                total_cost=1234.0,
                currency=currency,
                daily_plan=[{"day": 1, "activities": ["Welcome brunch", "City tour"]}],
                total_cost_usd=1234.0,
                fx_rates={"USD->USD": 1.0},
            )
            return PlanResponse(
                itinerary=itinerary,
                reasoning_trace=[
                    ReasoningStep(
                        step_number=1,
                        thought="Analyse constraints",
                        action="Draft itinerary",
                        observation="Constraints satisfied",
                    )
                ],
                tool_calls=[
                    ToolCall(
                        tool_name="search_attractions",
                        arguments={"city": destination},
                        result={"items_found": 3},
                        execution_time_ms=42.0,
                    )
                ],
                total_iterations=1,
                planning_time_ms=250.0,
                constraints_satisfied=True,
                constraint_violations=[],
                constraint_satisfaction=1.0,
                tool_errors_count=0,
                strategy_used={"name": "stub-strategy"},
                llm_token_usage={"prompt": 50, "completion": 25, "total": 75},
                llm_cost_usd=0.05,
            )

    agent = _DummyPlanningAgent()
    monkeypatch.setattr("backend.routers.agent_routes.get_planning_agent", lambda: agent)
    return agent


@pytest.fixture
def dummy_chat_service(monkeypatch):
    """Patch the chat service singleton with an in-memory stub."""
    from backend.models.chat_schemas import ChatMessage, ChatResponse

    class _DummyChatService:
        def __init__(self):
            self.messages: list[ChatMessage] = []
            self.model_name = "dummy-chat-model"

        async def chat_completion(self, user_message: str, max_history: int = 10) -> ChatResponse:
            self.messages.append(ChatMessage(role="user", content=user_message))
            reply = f"Echo: {user_message}"
            self.messages.append(ChatMessage(role="assistant", content=reply))
            return ChatResponse(
                message=reply,
                prompt_tokens=4,
                completion_tokens=4,
                total_tokens=8,
                cost_usd=0.0,
                latency_ms=0.5,
            )

        async def chat_completion_stream(  # pragma: no cover - exercised indirectly
            self, user_message: str, max_history: int = 10
        ) -> AsyncGenerator[str, None]:
            self.messages.append(ChatMessage(role="user", content=user_message))
            chunks = ["Echo", ": ", user_message]
            for chunk in chunks:
                yield chunk
            self.messages.append(ChatMessage(role="assistant", content="".join(chunks)))

        def get_history(self):
            return list(self.messages)

        def clear_history(self):
            self.messages.clear()

    service = _DummyChatService()
    monkeypatch.setattr("backend.routers.chat_routes.get_chat_service", lambda: service)
    return service
