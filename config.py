"""
Configuration management using Pydantic Settings.
Centralizes all application configuration with validation.
"""
import os
from functools import lru_cache
from typing import Optional
from schemas import AppConfig, ModelType, GenerationConfig


@lru_cache()
def get_app_config() -> AppConfig:
    """
    Get application configuration (cached).

    Returns:
        AppConfig: Application configuration instance
    """
    return AppConfig()


def get_api_key() -> Optional[str]:
    """
    Get the Google API key from environment or config.

    Returns:
        Optional[str]: API key if available
    """
    config = get_app_config()
    return config.google_api_key or os.getenv("GOOGLE_API_KEY")


def set_api_key(api_key: str) -> None:
    """
    Set the Google API key in environment.

    Args:
        api_key: The API key to set
    """
    os.environ["GOOGLE_API_KEY"] = api_key


def get_default_generation_config() -> GenerationConfig:
    """
    Get default generation configuration.

    Returns:
        GenerationConfig: Default generation configuration
    """
    config = get_app_config()
    return GenerationConfig(
        temperature=config.default_temperature,
        max_output_tokens=config.default_max_tokens
    )


def is_model_available(model_type: ModelType) -> bool:
    """
    Check if a specific model is available.

    Args:
        model_type: The model type to check

    Returns:
        bool: True if model is available
    """
    if model_type == ModelType.GEMINI_PRO:
        api_key = get_api_key()
        return api_key is not None and len(api_key) > 0

    # Other models don't require API keys
    return True


def validate_config() -> tuple[bool, Optional[str]]:
    """
    Validate the current configuration.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        config = get_app_config()

        # Validate temperature range
        if not 0.0 <= config.default_temperature <= 1.0:
            return False, "Temperature must be between 0.0 and 1.0"

        # Validate max tokens
        if config.default_max_tokens < 100:
            return False, "Max tokens must be at least 100"

        # Validate prompt history limit
        if config.max_prompt_history < 1:
            return False, "Max prompt history must be at least 1"

        return True, None

    except Exception as e:
        return False, f"Configuration validation error: {str(e)}"


def get_config_summary() -> dict:
    """
    Get a summary of the current configuration.

    Returns:
        dict: Configuration summary
    """
    config = get_app_config()
    api_key = get_api_key()

    return {
        "has_api_key": api_key is not None,
        "default_model": config.default_model.value,
        "default_temperature": config.default_temperature,
        "default_max_tokens": config.default_max_tokens,
        "code_validation_enabled": config.enable_code_validation,
        "code_formatting_enabled": config.enable_code_formatting,
        "max_prompt_history": config.max_prompt_history,
    }
