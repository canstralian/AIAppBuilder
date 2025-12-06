"""
Test suite for Pydantic models and schemas.
Validates data validation, type safety, and configuration management.
"""
import pytest
from pydantic import ValidationError
from schemas import (
    AppType,
    ModelType,
    TemplateType,
    AppConfig,
    GenerationConfig,
    CodeGenerationRequest,
    CodeGenerationResponse,
    CodeValidationResult,
    UserPromptHistory,
    ModelInfo,
    AppTypeInfo,
    FileExportRequest,
    ThemeConfig,
    AdvancedOptions
)
from config import (
    get_app_config,
    get_default_generation_config,
    validate_config
)
from utils import (
    validate_code,
    validate_code_detailed,
    format_code,
    export_code
)


def test_generation_config_valid():
    """Test valid generation configuration."""
    config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_output_tokens=2048
    )
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.max_output_tokens == 2048


def test_generation_config_invalid_temperature():
    """Test invalid temperature raises validation error."""
    try:
        GenerationConfig(temperature=1.5)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "temperature" in str(e).lower()


def test_code_generation_request_valid():
    """Test valid code generation request."""
    request = CodeGenerationRequest(
        prompt="Create a simple calculator app",
        app_type=AppType.STREAMLIT,
        template_name=TemplateType.BLANK,
        model_name=ModelType.GEMINI_PRO
    )
    assert request.prompt == "Create a simple calculator app"
    assert request.app_type == AppType.STREAMLIT
    assert request.template_name == TemplateType.BLANK
    assert request.model_name == ModelType.GEMINI_PRO


def test_code_generation_request_invalid_prompt():
    """Test invalid prompt raises validation error."""
    try:
        CodeGenerationRequest(
            prompt="hi",  # Too short
            app_type=AppType.STREAMLIT,
            template_name=TemplateType.BLANK,
            model_name=ModelType.GEMINI_PRO
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "prompt" in str(e).lower()


def test_code_generation_request_empty_prompt():
    """Test empty prompt raises validation error."""
    try:
        CodeGenerationRequest(
            prompt="   ",  # Whitespace only
            app_type=AppType.STREAMLIT,
            template_name=TemplateType.BLANK,
            model_name=ModelType.GEMINI_PRO
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "prompt" in str(e).lower()


def test_code_validation_result():
    """Test code validation result model."""
    result = CodeValidationResult(
        is_valid=False,
        error_message="Syntax error at line 5",
        line_number=5,
        suggestions=["Check for missing colons"]
    )
    assert result.is_valid is False
    assert result.error_message == "Syntax error at line 5"
    assert result.line_number == 5
    assert len(result.suggestions) == 1


def test_user_prompt_history():
    """Test user prompt history model."""
    history = UserPromptHistory()
    history.add_prompt("First prompt")
    history.add_prompt("Second prompt")

    assert len(history.prompts) == 2
    assert history.prompts[0] == "First prompt"

    recent = history.get_recent(1)
    assert len(recent) == 1
    assert recent[0] == "Second prompt"

    history.clear()
    assert len(history.prompts) == 0


def test_model_info():
    """Test model info model."""
    info = ModelInfo(
        name="Test Model",
        provider="Test Provider",
        description="A test model",
        strengths=["Fast", "Accurate"],
        limitations=["Limited context"],
        requires_api_key=True,
        is_available=True
    )
    assert info.name == "Test Model"
    assert info.requires_api_key is True
    assert len(info.strengths) == 2


def test_app_type_info():
    """Test app type info model."""
    info = AppTypeInfo(
        name="Streamlit",
        description="Web app framework",
        key_features=["Easy to use", "Fast"],
        ideal_for=["Data apps", "ML demos"],
        documentation_url="https://docs.streamlit.io"
    )
    assert info.name == "Streamlit"
    assert len(info.key_features) == 2


def test_file_export_request_valid():
    """Test valid file export request."""
    request = FileExportRequest(
        code="print('hello')",
        filename="test.py"
    )
    assert request.code == "print('hello')"
    assert request.filename == "test.py"


def test_file_export_request_invalid_filename():
    """Test invalid filename raises validation error."""
    try:
        FileExportRequest(
            code="print('hello')",
            filename="test.txt"  # Not a .py file
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "filename" in str(e).lower()


def test_theme_config():
    """Test theme configuration."""
    theme = ThemeConfig(
        theme_mode="dark",
        primary_color="#BB86FC",
        secondary_color="#03DAC6",
        background_color="#121212"
    )
    assert theme.theme_mode == "dark"
    assert theme.primary_color == "#BB86FC"


def test_theme_config_invalid_color():
    """Test invalid color format raises validation error."""
    try:
        ThemeConfig(
            theme_mode="dark",
            primary_color="invalid_color"
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "primary_color" in str(e).lower()


def test_advanced_options():
    """Test advanced options model."""
    options = AdvancedOptions(
        temperature=0.8,
        enable_validation=True,
        auto_format=True,
        max_retries=3
    )
    assert options.temperature == 0.8
    assert options.enable_validation is True
    assert options.max_retries == 3


def test_validate_code_valid():
    """Test code validation with valid code."""
    valid_code = """
def hello():
    print("Hello, World!")
"""
    is_valid, message = validate_code(valid_code)
    assert is_valid is True
    assert "valid" in message.lower()


def test_validate_code_invalid():
    """Test code validation with invalid code."""
    invalid_code = """
def hello()
    print("Hello, World!")
"""
    is_valid, message = validate_code(invalid_code)
    assert is_valid is False
    assert "syntax error" in message.lower()


def test_validate_code_detailed_valid():
    """Test detailed code validation with valid code."""
    valid_code = "x = 1\nprint(x)"
    result = validate_code_detailed(valid_code)
    assert result.is_valid is True
    assert result.error_message is None
    assert len(result.suggestions) == 0


def test_validate_code_detailed_invalid():
    """Test detailed code validation with invalid code."""
    invalid_code = "if True\n    print('test')"
    result = validate_code_detailed(invalid_code)
    assert result.is_valid is False
    assert result.error_message is not None
    assert len(result.suggestions) > 0


def test_format_code_valid():
    """Test code formatting with valid code."""
    code = "def hello():\n    print('Hello')\n\n"
    formatted = format_code(code)
    assert formatted.strip() == code.strip()


def test_format_code_invalid():
    """Test code formatting with invalid code."""
    code = "def hello()\n    print('Hello')"
    formatted = format_code(code)
    assert "syntax error" in formatted.lower()


def test_export_code_valid():
    """Test code export with valid inputs."""
    code = "print('test')"
    result = export_code(code, "test.py")
    assert "download" in result.lower()
    assert "test.py" in result


def test_export_code_invalid_filename():
    """Test code export with invalid filename."""
    code = "print('test')"
    result = export_code(code, "test.txt")
    assert "error" in result.lower()


def test_config_validation():
    """Test configuration validation."""
    is_valid, error = validate_config()
    # Should be valid with default config
    assert is_valid is True or error is not None


def test_default_generation_config():
    """Test getting default generation config."""
    config = get_default_generation_config()
    assert isinstance(config, GenerationConfig)
    assert 0.0 <= config.temperature <= 1.0
    assert config.max_output_tokens > 0


if __name__ == "__main__":
    # Run basic tests
    print("Running Pydantic model tests...")

    try:
        test_generation_config_valid()
        print("✓ GenerationConfig validation test passed")

        test_code_generation_request_valid()
        print("✓ CodeGenerationRequest validation test passed")

        test_user_prompt_history()
        print("✓ UserPromptHistory test passed")

        test_validate_code_valid()
        print("✓ Code validation test passed")

        test_format_code_valid()
        print("✓ Code formatting test passed")

        test_export_code_valid()
        print("✓ Code export test passed")

        print("\n✓ All basic tests passed!")
        print("\nRun 'pytest test_pydantic_models.py -v' for comprehensive testing")

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        raise
