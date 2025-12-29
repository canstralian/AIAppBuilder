"""
Pydantic models and schemas for data validation and type safety.
Provides comprehensive validation for configuration, requests, and responses.
"""
from typing import Optional, List, Dict, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class AppType(str, Enum):
    """Enumeration for supported application types."""
    STREAMLIT = "streamlit"
    GRADIO = "gradio"


class ModelType(str, Enum):
    """Enumeration for supported AI models."""
    GEMINI_PRO = "gemini_pro_20"
    CODET5 = "codet5"
    T0_3B = "t0_3b"


class TemplateType(str, Enum):
    """Enumeration for available templates."""
    BLANK = "blank"
    DATA_VIZ = "data_viz"
    FILE_UPLOAD = "file_upload"
    FORM = "form"
    NLP = "nlp"
    IMAGE_CLASSIFIER = "image_classifier"
    TEXT_GEN = "text_gen"
    AUDIO = "audio"
    CHAT = "chat"


class AppConfig(BaseSettings):
    """Application configuration using Pydantic Settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Keys
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for code generation"
    )

    # Model Configuration
    default_model: ModelType = Field(
        default=ModelType.CODET5,
        description="Default AI model to use"
    )

    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default temperature for generation"
    )

    default_max_tokens: int = Field(
        default=8192,
        ge=100,
        le=32000,
        description="Maximum tokens for generation"
    )

    # Application Settings
    enable_code_validation: bool = Field(
        default=True,
        description="Enable automatic code validation"
    )

    enable_code_formatting: bool = Field(
        default=True,
        description="Enable automatic code formatting"
    )

    max_prompt_history: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of prompts to keep in history"
    )


class GenerationConfig(BaseModel):
    """Configuration for code generation requests."""
    model_config = ConfigDict(frozen=False, validate_assignment=True)

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Controls randomness in generation"
    )

    top_p: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )

    top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Top-k sampling parameter"
    )

    max_output_tokens: int = Field(
        default=8192,
        ge=100,
        le=32000,
        description="Maximum tokens to generate"
    )

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within acceptable range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="User's description of the desired application"
    )

    app_type: AppType = Field(
        default=AppType.STREAMLIT,
        description="Type of application to generate"
    )

    template_name: TemplateType = Field(
        default=TemplateType.BLANK,
        description="Template to use as base"
    )

    model_name: ModelType = Field(
        default=ModelType.GEMINI_PRO,
        description="AI model to use for generation"
    )

    generation_config: Optional[GenerationConfig] = Field(
        default=None,
        description="Optional generation configuration"
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty and has meaningful content."""
        if not v or v.isspace():
            raise ValueError("Prompt cannot be empty or whitespace only")

        # Check for minimum word count
        word_count = len(v.split())
        if word_count < 3:
            raise ValueError("Prompt must contain at least 3 words")

        return v.strip()


class CodeGenerationResponse(BaseModel):
    """Response model for code generation."""
    model_config = ConfigDict(frozen=False)

    generated_code: str = Field(
        ...,
        description="The generated code"
    )

    is_valid: bool = Field(
        default=True,
        description="Whether the code passed validation"
    )

    validation_message: Optional[str] = Field(
        default=None,
        description="Validation error message if any"
    )

    model_used: ModelType = Field(
        ...,
        description="The model that generated the code"
    )

    template_used: TemplateType = Field(
        ...,
        description="The template that was used"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about generation"
    )


class CodeValidationResult(BaseModel):
    """Result of code validation."""
    is_valid: bool = Field(
        ...,
        description="Whether the code is valid"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if validation failed"
    )

    line_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Line number where error occurred"
    )

    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for fixing the error"
    )


class UserPromptHistory(BaseModel):
    """Model for storing user prompt history."""
    prompts: List[str] = Field(
        default_factory=list,
        max_length=100,
        description="List of user prompts"
    )

    def add_prompt(self, prompt: str) -> None:
        """Add a prompt to history."""
        if prompt and not prompt.isspace():
            self.prompts.append(prompt.strip())

    def get_recent(self, count: int = 10) -> List[str]:
        """Get recent prompts."""
        return self.prompts[-count:] if self.prompts else []

    def clear(self) -> None:
        """Clear all prompts."""
        self.prompts.clear()


class ModelInfo(BaseModel):
    """Information about an AI model."""
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider")
    description: str = Field(..., description="Model description")
    strengths: List[str] = Field(default_factory=list, description="Model strengths")
    limitations: List[str] = Field(default_factory=list, description="Model limitations")
    requires_api_key: bool = Field(default=False, description="Whether API key is required")
    is_available: bool = Field(default=True, description="Whether model is currently available")


class AppTypeInfo(BaseModel):
    """Information about an application type."""
    name: str = Field(..., description="App type name")
    description: str = Field(..., description="App type description")
    key_features: List[str] = Field(default_factory=list, description="Key features")
    ideal_for: List[str] = Field(default_factory=list, description="Ideal use cases")
    documentation_url: Optional[str] = Field(default=None, description="Documentation URL")


class FileExportRequest(BaseModel):
    """Request model for file export."""
    code: str = Field(
        ...,
        min_length=1,
        description="Code to export"
    )

    filename: str = Field(
        default="app.py",
        pattern=r"^[a-zA-Z0-9_\-]+\.py$",
        description="Filename for export"
    )

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename has proper Python extension."""
        if not v.endswith('.py'):
            raise ValueError("Filename must end with .py extension")
        return v


class ThemeConfig(BaseModel):
    """Configuration for application theme."""
    theme_mode: Literal["light", "dark"] = Field(
        default="light",
        description="Theme mode"
    )

    primary_color: str = Field(
        default="#BB86FC",
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Primary theme color"
    )

    secondary_color: str = Field(
        default="#03DAC6",
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Secondary theme color"
    )

    background_color: str = Field(
        default="#121212",
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Background color"
    )


class AdvancedOptions(BaseModel):
    """Advanced configuration options."""
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Generation temperature"
    )

    enable_validation: bool = Field(
        default=True,
        description="Enable code validation"
    )

    auto_format: bool = Field(
        default=True,
        description="Automatically format generated code"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for generation"
    )
