"""
Simple test to validate Pydantic schemas without external dependencies.
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
    AppTypeInfo,
    FileExportRequest,
    ThemeConfig,
    AdvancedOptions
)
from config import get_default_generation_config, validate_config


def main():
    """Run validation tests."""
    print("=" * 60)
    print("Pydantic Schema Validation Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: GenerationConfig
    print("\n1. Testing GenerationConfig...")
    try:
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_output_tokens=2048
        )
        print(f"   ✓ Created: temperature={config.temperature}, max_tokens={config.max_output_tokens}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 2: Invalid temperature
    print("\n2. Testing invalid temperature (should fail)...")
    try:
        GenerationConfig(temperature=1.5)
        print("   ✗ Should have raised ValidationError")
        all_passed = False
    except ValidationError as e:
        print("   ✓ Correctly rejected invalid temperature")

    # Test 3: CodeGenerationRequest
    print("\n3. Testing CodeGenerationRequest...")
    try:
        request = CodeGenerationRequest(
            prompt="Create a data visualization app with charts and graphs",
            app_type=AppType.STREAMLIT,
            template_name=TemplateType.DATA_VIZ,
            model_name=ModelType.GEMINI_PRO
        )
        print(f"   ✓ Created: app_type={request.app_type.value}, model={request.model_name.value}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 4: Invalid prompt (too short)
    print("\n4. Testing invalid prompt (should fail)...")
    try:
        CodeGenerationRequest(
            prompt="hi",
            app_type=AppType.STREAMLIT,
            template_name=TemplateType.BLANK,
            model_name=ModelType.CODET5
        )
        print("   ✗ Should have raised ValidationError")
        all_passed = False
    except ValidationError as e:
        print("   ✓ Correctly rejected short prompt")

    # Test 5: UserPromptHistory
    print("\n5. Testing UserPromptHistory...")
    try:
        history = UserPromptHistory()
        history.add_prompt("First prompt for testing")
        history.add_prompt("Second prompt for testing")
        recent = history.get_recent(1)
        assert len(recent) == 1
        assert recent[0] == "Second prompt for testing"
        print(f"   ✓ Created and managed history: {len(history.prompts)} prompts")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 6: FileExportRequest
    print("\n6. Testing FileExportRequest...")
    try:
        request = FileExportRequest(
            code="print('Hello, World!')",
            filename="hello.py"
        )
        print(f"   ✓ Created: filename={request.filename}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 7: Invalid filename (should fail)
    print("\n7. Testing invalid filename (should fail)...")
    try:
        FileExportRequest(
            code="print('test')",
            filename="test.txt"
        )
        print("   ✗ Should have raised ValidationError")
        all_passed = False
    except ValidationError as e:
        print("   ✓ Correctly rejected non-.py filename")

    # Test 8: ModelInfo
    print("\n8. Testing ModelInfo...")
    try:
        info = ModelInfo(
            name="Gemini Pro",
            provider="Google",
            description="Advanced AI model",
            strengths=["Fast", "Accurate", "Versatile"],
            limitations=["Requires API key"],
            requires_api_key=True,
            is_available=True
        )
        print(f"   ✓ Created: {info.name} by {info.provider}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 9: AppTypeInfo
    print("\n9. Testing AppTypeInfo...")
    try:
        info = AppTypeInfo(
            name="Streamlit",
            description="Python web app framework",
            key_features=["Fast", "Easy", "Pythonic"],
            ideal_for=["Data apps", "ML demos"],
            documentation_url="https://docs.streamlit.io"
        )
        print(f"   ✓ Created: {info.name} with {len(info.key_features)} features")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 10: ThemeConfig
    print("\n10. Testing ThemeConfig...")
    try:
        theme = ThemeConfig(
            theme_mode="dark",
            primary_color="#BB86FC",
            secondary_color="#03DAC6",
            background_color="#121212"
        )
        print(f"   ✓ Created: mode={theme.theme_mode}, primary={theme.primary_color}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 11: AdvancedOptions
    print("\n11. Testing AdvancedOptions...")
    try:
        options = AdvancedOptions(
            temperature=0.8,
            enable_validation=True,
            auto_format=True,
            max_retries=3
        )
        print(f"   ✓ Created: temp={options.temperature}, retries={options.max_retries}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 12: CodeValidationResult
    print("\n12. Testing CodeValidationResult...")
    try:
        result = CodeValidationResult(
            is_valid=False,
            error_message="Syntax error",
            line_number=42,
            suggestions=["Check indentation", "Add missing colon"]
        )
        print(f"   ✓ Created: valid={result.is_valid}, suggestions={len(result.suggestions)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 13: Default generation config
    print("\n13. Testing default generation config...")
    try:
        config = get_default_generation_config()
        assert isinstance(config, GenerationConfig)
        print(f"   ✓ Retrieved: temp={config.temperature}, tokens={config.max_output_tokens}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Test 14: Config validation
    print("\n14. Testing config validation...")
    try:
        is_valid, error = validate_config()
        print(f"   ✓ Validated: is_valid={is_valid}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nPydantic Integration Summary:")
        print("  ✓ Data validation is working correctly")
        print("  ✓ Type safety is enforced")
        print("  ✓ Configuration management is operational")
        print("  ✓ All Pydantic models are properly defined")
        print("  ✓ Enum types are correctly implemented")
        print("  ✓ Field validators are functioning")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
