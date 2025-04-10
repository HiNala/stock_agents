from typing import Dict, Any, Optional
import os
import openai
import anthropic
from huggingface_hub import InferenceClient
from src.config.model_config import ModelProvider, model_config

class BaseLLMClient:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = model_config.get_agent_config(agent_name)
        self.provider = self.config["provider"]
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        
        # Initialize provider-specific clients
        self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Initialize the appropriate LLM client based on the provider."""
        provider_config = model_config.get_provider_config(self.provider)
        
        if self.provider == ModelProvider.OPENAI:
            api_key = os.getenv(provider_config["api_key_env_var"])
            if not api_key:
                raise ValueError(f"Missing {provider_config['api_key_env_var']} environment variable")
            openai.api_key = api_key
            self.client = openai
            
        elif self.provider == ModelProvider.ANTHROPIC:
            api_key = os.getenv(provider_config["api_key_env_var"])
            if not api_key:
                raise ValueError(f"Missing {provider_config['api_key_env_var']} environment variable")
            self.client = anthropic.Anthropic(api_key=api_key)
            
        elif self.provider == ModelProvider.HUGGINGFACE:
            api_key = os.getenv(provider_config["api_key_env_var"])
            if not api_key:
                raise ValueError(f"Missing {provider_config['api_key_env_var']} environment variable")
            self.client = InferenceClient(model=self.model, token=api_key)
            
        elif self.provider == ModelProvider.LOCAL:
            # Implement local model initialization here
            pass
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        if self.provider == ModelProvider.OPENAI:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
            
        elif self.provider == ModelProvider.ANTHROPIC:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text
            
        elif self.provider == ModelProvider.HUGGINGFACE:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            return response
            
        elif self.provider == ModelProvider.LOCAL:
            # Implement local model generation here
            pass
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the client's configuration."""
        model_config.update_agent_config(self.agent_name, new_config)
        self.config = model_config.get_agent_config(self.agent_name)
        self.provider = self.config["provider"]
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self._initialize_provider() 