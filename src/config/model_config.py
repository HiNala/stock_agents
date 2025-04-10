from typing import Dict, Any
from enum import Enum
from src.config.settings import (
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    AGENT_SETTINGS
)

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class ModelConfig:
    def __init__(self):
        # Default model configurations
        self.default_provider = ModelProvider(DEFAULT_MODEL_PROVIDER)
        self.default_model = DEFAULT_MODEL
        self.default_temperature = DEFAULT_TEMPERATURE
        self.default_max_tokens = DEFAULT_MAX_TOKENS
        
        # Provider-specific configurations
        self.provider_configs: Dict[ModelProvider, Dict[str, Any]] = {
            ModelProvider.OPENAI: {
                "api_key_env_var": "OPENAI_API_KEY",
                "default_model": DEFAULT_MODEL,
                "available_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            },
            ModelProvider.ANTHROPIC: {
                "api_key_env_var": "ANTHROPIC_API_KEY",
                "default_model": "claude-3-opus-20240229",
                "available_models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            },
            ModelProvider.HUGGINGFACE: {
                "api_key_env_var": "HUGGINGFACE_API_KEY",
                "default_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "available_models": ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf"],
            },
            ModelProvider.LOCAL: {
                "default_model": "local-model",
                "available_models": ["local-model"],
            }
        }
        
        # Agent-specific model configurations
        self.agent_configs: Dict[str, Dict[str, Any]] = {
            "research_agent": {
                "provider": ModelProvider.OPENAI,
                "model": AGENT_SETTINGS["research_agent"]["model"],
                "temperature": AGENT_SETTINGS["research_agent"]["temperature"],
                "max_tokens": AGENT_SETTINGS["research_agent"]["max_tokens"],
            },
            "universe_agent": {
                "provider": ModelProvider.OPENAI,
                "model": AGENT_SETTINGS["universe_agent"]["model"],
                "temperature": AGENT_SETTINGS["universe_agent"]["temperature"],
                "max_tokens": AGENT_SETTINGS["universe_agent"]["max_tokens"],
            },
            "strategy_agent": {
                "provider": ModelProvider.OPENAI,
                "model": AGENT_SETTINGS["strategy_agent"]["model"],
                "temperature": AGENT_SETTINGS["strategy_agent"]["temperature"],
                "max_tokens": AGENT_SETTINGS["strategy_agent"]["max_tokens"],
            },
            "risk_agent": {
                "provider": ModelProvider.OPENAI,
                "model": AGENT_SETTINGS["risk_agent"]["model"],
                "temperature": AGENT_SETTINGS["risk_agent"]["temperature"],
                "max_tokens": AGENT_SETTINGS["risk_agent"]["max_tokens"],
            },
            "play_agent": {
                "provider": ModelProvider.OPENAI,
                "model": AGENT_SETTINGS["play_agent"]["model"],
                "temperature": AGENT_SETTINGS["play_agent"]["temperature"],
                "max_tokens": AGENT_SETTINGS["play_agent"]["max_tokens"],
            }
        }

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get the model configuration for a specific agent."""
        if agent_name not in self.agent_configs:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self.agent_configs[agent_name]

    def get_provider_config(self, provider: ModelProvider) -> Dict[str, Any]:
        """Get the configuration for a specific provider."""
        if provider not in self.provider_configs:
            raise ValueError(f"Unknown provider: {provider}")
        return self.provider_configs[provider]

    def update_agent_config(self, agent_name: str, config: Dict[str, Any]) -> None:
        """Update the configuration for a specific agent."""
        if agent_name not in self.agent_configs:
            raise ValueError(f"Unknown agent: {agent_name}")
        self.agent_configs[agent_name].update(config)

    def update_provider_config(self, provider: ModelProvider, config: Dict[str, Any]) -> None:
        """Update the configuration for a specific provider."""
        if provider not in self.provider_configs:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider_configs[provider].update(config)

# Create a singleton instance
model_config = ModelConfig() 