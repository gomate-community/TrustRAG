# from pydantic import BaseSettings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    """Centralized application settings loaded from environment variables."""

    # API service
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=10000, validation_alias="API_PORT")
    api_url: str = Field(default="http://127.0.0.1:10001", validation_alias="API_URL")

    # Upstream LLM service (gomall)
    gomall_base_url: str = Field(default="http://10.208.61.1:32004/api/v1/1504_gomall_qwen3/Qwen3-30B-A3B-Instruct-2507/", validation_alias="GOMALL_BASE_URL")
    gomall_api_key: str = Field(default="", validation_alias="GOMALL_API_KEY")
    llm_name: str = Field(default="Qwen3-30B-A3B-Instruct-2507", validation_alias="LLM_NAME")
    # Tool specific
    rewriter_api_url: str = Field(default="http://10.208.61.1:32004/api/v1/1504_gomall_qwen3/Qwen3-30B-A3B-Instruct-2507/", validation_alias="REWRITER_API_URL")

    # Local rerank service
    rerank_base_url: str | None = Field(default=None, validation_alias="RERANK_BASE_URL")
    rerank_api_key: str | None = Field(default=None, validation_alias="RERANK_API_KEY")


    embedding_name: str = Field(default="G:/pretrained_models/mteb/bge-large-zh-v1.5", validation_alias="EMBEDDING_NAME")
    embedding_path: str = Field(default="G:/pretrained_models/mteb/bge-large-zh-v1.5", validation_alias="EMBEDDING_PATH")

    reranker_name: str = Field(default="G:/pretrained_models/mteb/bge-reranker-large", validation_alias="RERANKER_NAME")
    reranker_path: str = Field(default="G:/pretrained_models/mteb/bge-reranker-large", validation_alias="RERANKER_PATH")
    # Paths
    docs_path: str = Field(default="G:/Projects/TrustRAG/data/docs", validation_alias="DOCS_PATH")
    index_path: str = Field(default="G:/Projects/TrustRAG/examples/retrievers/dense_cache", validation_alias="INDEX_PATH")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )