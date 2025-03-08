import os
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

from rich.console import Console

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_OLLAMA_MODEL = None


class ServiceProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""

    service_provider_name: str
    api_key_env: str
    default_url: str
    url_env_var: str
    default_model: Optional[str] = None
    model_env_var: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        """Get the API key from environment variables."""
        if self.service_provider_name == ServiceProvider.OLLAMA.value:
            return (
                "no_api_key"  # Have to send some string to the client to not get error
            )
        return os.getenv(self.api_key_env)

    @property
    def base_url(self) -> str:
        """Get the base URL, with fallback to default."""
        return os.getenv(self.url_env_var, self.default_url)

    @property
    def model(self) -> Optional[str]:
        """Get the model name, with fallback to default if configured."""
        if self.model_env_var:
            return os.getenv(self.model_env_var, self.default_model)
        return self.default_model


class EnvironmentConfig:
    """Central configuration management for AI providers."""

    # Define all supported providers with their configurations
    PROVIDERS: Dict[str, ProviderConfig] = {
        ServiceProvider.OPENAI.value: ProviderConfig(
            service_provider_name=ServiceProvider.OPENAI.value,
            api_key_env="OPENAI_API_KEY",
            # default_url="https://api.openai.com/v1",
            default_url="https://www.dmxapi.com/v1",
            url_env_var="OPENAI_API_ENDPOINT",
            default_model=DEFAULT_OPENAI_MODEL,
            model_env_var="OPENAI_MODEL",
        ),
        ServiceProvider.DEEPSEEK.value: ProviderConfig(
            service_provider_name=ServiceProvider.DEEPSEEK.value,
            api_key_env="DEEPSEEK_API_KEY",
            default_url="https://api.deepseek.com/v1",
            url_env_var="DEEPSEEK_API_ENDPOINT",
            default_model=DEFAULT_DEEPSEEK_MODEL,
            model_env_var="DEEPSEEK_MODEL",
        ),
        ServiceProvider.OLLAMA.value: ProviderConfig(
            service_provider_name=ServiceProvider.OLLAMA.value,
            api_key_env="OLLAMA_API_KEY",
            default_url="http://localhost:11434/v1",
            url_env_var="OLLAMA_HOST_ENDPOINT",
            default_model=DEFAULT_OLLAMA_MODEL,
            model_env_var="OLLAMA_MODEL",
        ),
    }

    @classmethod
    def get_default_provider(cls) -> str:
        """Get the default provider name from environment."""
        return os.getenv("DEFAULT_SERVICE", "").lower()

    @classmethod
    def get_provider_config(
        cls, service_provider_name: Optional[str] = None
    ) -> ProviderConfig:
        """Get configuration for a specific provider or the default provider."""
        provider = service_provider_name or cls.get_default_provider()

        if not provider:
            raise ValueError(
                "No provider specified and DEFAULT_SERVICE not set in environment"
            )

        if provider not in cls.PROVIDERS:
            supported = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(f"Invalid provider '{provider}'. Choose from: {supported}")

        return cls.PROVIDERS[provider]

    @classmethod
    def validate_provider_config(
        cls,
        service_provider_name: Optional[str] = None,
        console: Optional[Console] = None,
    ) -> ProviderConfig:
        """Validate the provider configuration and return it if valid."""
        console = console or Console()

        try:
            config = cls.get_provider_config(service_provider_name)

            if not config.api_key:
                console.print(f"[red]Missing {config.api_key_env} in environment[/red]")
                raise ValueError(f"Missing {config.api_key_env}")

            if not config.model:
                console.print(
                    f"[yellow]Warning: No model specified for {config.service_provider_name}[/yellow]"
                )

            return config

        except ValueError as e:
            console.print(f"[red]{str(e)}[/red]")
            raise