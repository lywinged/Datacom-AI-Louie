from prometheus_client import Counter, Gauge, Histogram

role_profile_base_sync_counter = Counter(
    "role_profile_base_sync_total",
    "Count of role base profile synchronization outcomes",
    ["action"],
)

role_profile_diary_sync_counter = Counter(
    "role_profile_diary_sync_total",
    "Count of role diary synchronization outcomes",
    ["action"],
)

role_profile_section_sync_counter = Counter(
    "role_profile_section_sync_total",
    "Count of role section synchronization outcomes",
    ["action"],
)

# Token usage metrics
llm_token_usage_counter = Counter(
    "llm_token_usage_total",
    "Total tokens consumed by LLM calls",
    ["model", "token_type"],  # token_type: prompt|completion
)

llm_request_counter = Counter(
    "llm_request_total",
    "Total number of LLM API requests",
    ["model", "endpoint", "status"],  # endpoint: chat|embedding|etc, status: success|error
)

llm_cost_counter = Counter(
    "llm_cost_usd_total",
    "Estimated total cost in USD for LLM usage",
    ["model"],
)

llm_request_duration_histogram = Histogram(
    "llm_request_duration_seconds",
    "Duration of LLM API requests in seconds",
    ["model", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Cold/Hot tier metrics
knowledge_tier_operations_counter = Counter(
    "knowledge_tier_operations_total",
    "Count of cold/hot tier operations",
    ["operation", "outcome"],  # operation: archive|rehydrate, outcome: success|error
)

knowledge_cold_documents_gauge = Gauge(
    "knowledge_cold_documents",
    "Current number of cold documents",
)

knowledge_hot_documents_gauge = Gauge(
    "knowledge_hot_documents",
    "Current number of hot documents",
)

knowledge_cold_chunks_gauge = Gauge(
    "knowledge_cold_chunks",
    "Current number of chunks with null embeddings (cold)",
)

# Initialize counters with zero values so they appear in metrics immediately
# This makes them visible in Prometheus even before any events occur
def _initialize_metrics():
    """Initialize all metrics with default labels to make them visible"""
    role_profile_base_sync_counter.labels(action="success")._value.set(0)
    role_profile_base_sync_counter.labels(action="error")._value.set(0)
    role_profile_base_sync_counter.labels(action="embedded")._value.set(0)
    role_profile_diary_sync_counter.labels(action="success")._value.set(0)
    role_profile_diary_sync_counter.labels(action="error")._value.set(0)
    role_profile_diary_sync_counter.labels(action="embedded")._value.set(0)
    role_profile_section_sync_counter.labels(action="success")._value.set(0)
    role_profile_section_sync_counter.labels(action="error")._value.set(0)
    role_profile_section_sync_counter.labels(action="embedded")._value.set(0)

    # Initialize tier operations counter
    knowledge_tier_operations_counter.labels(operation="archive", outcome="success")._value.set(0)
    knowledge_tier_operations_counter.labels(operation="archive", outcome="error")._value.set(0)
    knowledge_tier_operations_counter.labels(operation="rehydrate", outcome="success")._value.set(0)
    knowledge_tier_operations_counter.labels(operation="rehydrate", outcome="error")._value.set(0)

    # Initialize gauges to 0
    knowledge_cold_documents_gauge.set(0)
    knowledge_hot_documents_gauge.set(0)
    knowledge_cold_chunks_gauge.set(0)

    # Initialize LLM metrics with common models
    common_models = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-5-sonnet", "deepseek-chat", "deepseek-v3", "deepseek-v3-250324", "deepseek-v3-1-terminus"]
    common_endpoints = ["chat", "story_chat", "embedding"]
    for model in common_models:
        llm_token_usage_counter.labels(model=model, token_type="prompt")._value.set(0)
        llm_token_usage_counter.labels(model=model, token_type="completion")._value.set(0)
        for endpoint in common_endpoints:
            llm_request_counter.labels(model=model, endpoint=endpoint, status="success")._value.set(0)
            llm_request_counter.labels(model=model, endpoint=endpoint, status="error")._value.set(0)
        llm_cost_counter.labels(model=model)._value.set(0)

_initialize_metrics()

# === Inference Service Metrics (Optional, only if enabled) ===
inference_request_counter = Counter(
    "inference_request_total",
    "Total number of inference requests",
    ["service", "status"]  # service: embedding/rerank, status: success/error/fallback
)

inference_latency_histogram = Histogram(
    "inference_request_duration_seconds",
    "Inference request duration in seconds",
    ["service"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

inference_error_counter = Counter(
    "inference_error_total",
    "Total number of inference errors",
    ["service", "error_type"]  # error_type: timeout/connection/http_xxx/circuit_breaker_open
)

circuit_breaker_state_gauge = Gauge(
    "inference_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open)",
    ["service"]
)

# RAG Operation Metrics
rag_operation_counter = Counter(
    "rag_operation_total",
    "Total number of RAG operations",
    ["operation", "source"]  # operation: embed/rerank/search, source: remote/local/fallback
)

rag_latency_histogram = Histogram(
    "rag_operation_duration_seconds",
    "RAG operation duration in seconds",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# === Enhanced RAG Performance Metrics ===

# pgvector query performance
pgvector_query_duration_histogram = Histogram(
    "pgvector_query_duration_seconds",
    "Duration of pgvector similarity search queries",
    ["query_type", "role_filtered"],  # query_type: vector|text|hybrid, role_filtered: yes|no
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

pgvector_query_counter = Counter(
    "pgvector_query_total",
    "Total number of pgvector queries",
    ["query_type", "status"]  # status: success|error|fallback
)

pgvector_candidates_returned_histogram = Histogram(
    "pgvector_candidates_returned",
    "Number of candidates returned by pgvector query",
    ["query_type"],
    buckets=[0, 1, 5, 10, 20, 50, 100, 200, 500]
)

# Embedding performance (local model)
embedding_duration_histogram = Histogram(
    "embedding_duration_seconds",
    "Duration of embedding generation",
    ["model", "source", "batch_size_range"],  # source: local|remote|fallback, batch_size_range: 1|2-10|11-50|50+
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

embedding_counter = Counter(
    "embedding_total",
    "Total number of embedding operations",
    ["model", "source", "status"]  # status: success|error|cache_hit
)

embedding_tokens_histogram = Histogram(
    "embedding_tokens",
    "Number of tokens processed in embedding",
    ["model"],
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
)

# Reranking performance
rerank_duration_histogram = Histogram(
    "rerank_duration_seconds",
    "Duration of reranking operation",
    ["model", "source", "candidate_count_range"],  # source: local|remote, candidate_count_range: 1-10|11-50|50+
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

rerank_counter = Counter(
    "rerank_total",
    "Total number of reranking operations",
    ["model", "source", "status"]  # status: success|error|skipped
)

rerank_score_distribution_histogram = Histogram(
    "rerank_score_distribution",
    "Distribution of reranking scores",
    ["model"],
    buckets=[-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# End-to-end RAG request performance
rag_request_duration_histogram = Histogram(
    "rag_request_duration_seconds",
    "End-to-end RAG request duration",
    ["endpoint", "has_role_filter"],  # endpoint: story_chat|chat|search
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
)

rag_request_counter = Counter(
    "rag_request_total",
    "Total number of RAG requests",
    ["endpoint", "status"]  # status: success|error|partial
)

# Model version tracking
model_info_gauge = Gauge(
    "model_version_info",
    "Model version information (value=1 when model loaded)",
    ["model_type", "model_name", "model_version", "model_hash"]  # model_type: embedding|reranker
)

# Initialize circuit-breaker state gauges
circuit_breaker_state_gauge.labels(service="embedding").set(0)
circuit_breaker_state_gauge.labels(service="rerank").set(0)

__all__ = [
    "role_profile_base_sync_counter",
    "role_profile_diary_sync_counter",
    "role_profile_section_sync_counter",
    "knowledge_tier_operations_counter",
    "knowledge_cold_documents_gauge",
    "knowledge_hot_documents_gauge",
    "knowledge_cold_chunks_gauge",
    "llm_token_usage_counter",
    "llm_request_counter",
    "llm_cost_counter",
    "llm_request_duration_histogram",
    # Inference metrics
    "inference_request_counter",
    "inference_latency_histogram",
    "inference_error_counter",
    "circuit_breaker_state_gauge",
    "rag_operation_counter",
    "rag_latency_histogram",
    # Enhanced RAG metrics
    "pgvector_query_duration_histogram",
    "pgvector_query_counter",
    "pgvector_candidates_returned_histogram",
    "embedding_duration_histogram",
    "embedding_counter",
    "embedding_tokens_histogram",
    "rerank_duration_histogram",
    "rerank_counter",
    "rerank_score_distribution_histogram",
    "rag_request_duration_histogram",
    "rag_request_counter",
    "model_info_gauge",
]
