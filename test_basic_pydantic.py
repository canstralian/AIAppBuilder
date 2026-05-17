"""
Basic validation tests for Pydantic models without pytest dependency.
"""
from pydantic import ValidationError
from schemas import (
    AppType,
    ModelType,
    TemplateType,
    GenerationConfig,
    CodeGenerationRequest,
    CodeValidationResult,
    UserPromptHistory,
    ModelInfo,
    FileExportRequest,
)
from config import get_default_generation_config, validate_config
from utils import validate_code, validate_code_detailed, format_code


def test_generation_config():
    """Test generation configuration."""
    print("Testing GenerationConfig...")
    config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_output_tokens=2048
    )
    assert config.temperature == 0.7
    print("  ✓ GenerationConfig created successfully")

    # Test invalid temperature
    try:
        GenerationConfig(temperature=1.5)
        print("  ✗ Should have raised ValidationError for invalid temperature")
        return False
    except ValidationError:
        print("  ✓ Validation error correctly raised for invalid temperature")

    return True


def test_code_generation_request():
    """Test code generation request."""
    print("\nTesting CodeGenerationRequest...")
    request = CodeGenerationRequest(
        prompt="Create a simple calculator app with basic operations",
        app_type=AppType.STREAMLIT,
        template_name=TemplateType.BLANK,
        model_name=ModelType.GEMINI_PRO
    )
    assert request.prompt == "Create a simple calculator app with basic operations"
    print("  ✓ CodeGenerationRequest created successfully")

    # Test invalid prompt (too short)
    try:
        CodeGenerationRequest(
            prompt="hi",
            app_type=AppType.STREAMLIT,
            template_name=TemplateType.BLANK,
            model_name=ModelType.GEMINI_PRO
        )
        print("  ✗ Should have raised ValidationError for short prompt")
        return False
    except ValidationError:
        print("  ✓ Validation error correctly raised for short prompt")

    return True


def test_code_validation():
    """Test code validation."""
    print("\nTesting code validation...")

    # Valid code
    valid_code = "def hello():\n    print('Hello, World!')"
    is_valid, message = validate_code(valid_code)
    assert is_valid is True
    print("  ✓ Valid code recognized correctly")

    # Invalid code
    invalid_code = "def hello()\n    print('Hello')"
    is_valid, message = validate_code(invalid_code)
    assert is_valid is False
    print("  ✓ Invalid code detected correctly")

    return True


def test_code_validation_detailed():
    """Test detailed code validation."""
    print("\nTesting detailed code validation...")

    valid_code = "x = 1\nprint(x)"
    result = validate_code_detailed(valid_code)
    assert result.is_valid is True
    assert result.error_message is None
    print("  ✓ Detailed validation works for valid code")

    invalid_code = "if True\n    print('test')"
    result = validate_code_detailed(invalid_code)
    assert result.is_valid is False
    assert result.error_message is not None
    assert len(result.suggestions) > 0
    print("  ✓ Detailed validation provides suggestions for invalid code")

    return True


def test_user_prompt_history():
    """Test user prompt history."""
    print("\nTesting UserPromptHistory...")
    history = UserPromptHistory()
    history.add_prompt("First prompt")
    history.add_prompt("Second prompt")

    assert len(history.prompts) == 2
    print("  ✓ Prompts added correctly")

    recent = history.get_recent(1)
    assert len(recent) == 1
    assert recent[0] == "Second prompt"
    print("  ✓ Recent prompts retrieved correctly")

    history.clear()
    assert len(history.prompts) == 0
    print("  ✓ History cleared successfully")

    return True


def test_file_export_request():
    """Test file export request."""
    print("\nTesting FileExportRequest...")

    request = FileExportRequest(
        code="print('hello')",
        filename="test.py"
    )
    assert request.filename == "test.py"
    print("  ✓ Valid FileExportRequest created")

    # Test invalid filename
    try:
        FileExportRequest(
            code="print('hello')",
            filename="test.txt"
        )
        print("  ✗ Should have raised ValidationError for invalid filename")
        return False
    except ValidationError:
        print("  ✓ Validation error correctly raised for invalid filename")

    return True


def test_model_info():
    """Test model info."""
    print("\nTesting ModelInfo...")

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
    assert len(info.strengths) == 2
    print("  ✓ ModelInfo created successfully")

    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    config = get_default_generation_config()
    assert isinstance(config, GenerationConfig)
    assert 0.0 <= config.temperature <= 1.0
    print("  ✓ Default generation config retrieved")

    is_valid, error = validate_config()
    print(f"  ✓ Config validation: {is_valid}, Error: {error}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Basic Pydantic Model Tests")
    print("=" * 60)

    tests = [
        test_generation_config,
        test_code_generation_request,
        test_code_validation,
        test_code_validation_detailed,
        test_user_prompt_history,
        test_file_export_request,
        test_model_info,
        test_config,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All tests passed successfully!")
        print("\nPydantic integration is working correctly:")
        print("  - Data validation is functioning")
        print("  - Type safety is enforced")
        print("  - Configuration management is operational")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
