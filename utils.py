"""
Utility functions for code validation, formatting, and export.
Enhanced with Pydantic validation for type safety and data integrity.
"""
import streamlit as st
import ast
import base64
from typing import Tuple, Optional
from schemas import (
    CodeValidationResult,
    FileExportRequest,
    ModelInfo,
    AppTypeInfo,
    ModelType
)

def format_code(code: str) -> str:
    """Format the generated code with syntax validation.

    Args:
        code: The code to format

    Returns:
        str: The formatted code
    """
    if not code or code.isspace():
        return "# Empty code provided"

    try:
        # Parse the code to validate syntax
        ast.parse(code)
        # Code is valid, return as-is (can add formatter like black here)
        return code.strip()
    except SyntaxError as e:
        # If there's a syntax error, add a comment about the error
        error_line = e.lineno if e.lineno else "unknown"
        return f"# Note: Syntax error detected at line {error_line}\n# {str(e)}\n\n{code}"
    except Exception as e:
        # Handle other parsing errors
        return f"# Note: Error during code formatting: {str(e)}\n\n{code}"

def validate_code(code: str) -> Tuple[bool, str]:
    """Validate if the code has proper syntax.

    Args:
        code: The code to validate

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not code or code.isspace():
        return False, "Code is empty or contains only whitespace"

    try:
        ast.parse(code)
        return True, "Code syntax is valid."
    except SyntaxError as e:
        line_number = e.lineno if e.lineno else 0
        error_message = str(e)
        return False, f"Syntax error at line {line_number}: {error_message}"
    except Exception as e:
        return False, f"Error validating code: {str(e)}"


def validate_code_detailed(code: str) -> CodeValidationResult:
    """Validate code and return detailed validation result.

    Args:
        code: The code to validate

    Returns:
        CodeValidationResult: Detailed validation result with suggestions
    """
    if not code or code.isspace():
        return CodeValidationResult(
            is_valid=False,
            error_message="Code is empty or contains only whitespace",
            suggestions=["Provide valid Python code"]
        )

    try:
        ast.parse(code)
        return CodeValidationResult(
            is_valid=True,
            error_message=None,
            suggestions=[]
        )
    except SyntaxError as e:
        suggestions = []
        if "invalid syntax" in str(e).lower():
            suggestions.append("Check for missing colons, parentheses, or brackets")
        if "indentation" in str(e).lower():
            suggestions.append("Verify consistent indentation (use spaces, not tabs)")

        return CodeValidationResult(
            is_valid=False,
            error_message=str(e),
            line_number=e.lineno,
            suggestions=suggestions or ["Review Python syntax documentation"]
        )
    except Exception as e:
        return CodeValidationResult(
            is_valid=False,
            error_message=f"Validation error: {str(e)}",
            suggestions=["Check for unusual characters or encoding issues"]
        )

def export_code(code: str, filename: str = "app.py") -> str:
    """Create a download link for the code file with validation.

    Args:
        code: The code to export
        filename: The name of the file (defaults to "app.py")

    Returns:
        str: HTML code for a download link

    Raises:
        ValueError: If the code is empty or filename is invalid
    """
    try:
        # Validate using Pydantic model
        export_request = FileExportRequest(code=code, filename=filename)

        # Encode the code
        b64 = base64.b64encode(export_request.code.encode()).decode()
        href = f'<a href="data:file/text;base64,{b64}" download="{export_request.filename}" class="download-btn">💾 Download {export_request.filename}</a>'
        return href
    except Exception as e:
        # Return an error message if validation fails
        return f'<span style="color: red;">Error creating download link: {str(e)}</span>'

def get_app_type_info(app_type: str) -> str:
    """Return information about app types.

    Args:
        app_type: The type of app

    Returns:
        str: Formatted information about the app type
    """
    info_models = {
        "streamlit": AppTypeInfo(
            name="Streamlit",
            description="Open-source Python framework for creating beautiful web apps for ML and data science",
            key_features=[
                "Rapid prototyping",
                "Simple Python API",
                "Real-time updates",
                "Interactive widgets",
                "Easy deployment"
            ],
            ideal_for=[
                "Data visualization dashboards",
                "Machine learning demonstrations",
                "Simple web tools",
                "Data exploration apps"
            ],
            documentation_url="https://docs.streamlit.io"
        ),
        "gradio": AppTypeInfo(
            name="Gradio",
            description="Python library for creating customizable UI components for ML models and APIs",
            key_features=[
                "Easy interface creation",
                "Multiple input/output types",
                "Simplified deployment",
                "API generation",
                "Hugging Face integration"
            ],
            ideal_for=[
                "ML model demos",
                "Image/audio/text processing interfaces",
                "Multi-modal applications",
                "Interactive machine learning demos"
            ],
            documentation_url="https://www.gradio.app/docs"
        )
    }

    app_info = info_models.get(app_type.lower())
    if not app_info:
        return "No information available for this app type."

    # Format the information
    return f"""
### {app_info.name}

**Description:** {app_info.description}

**Key Features:**
{chr(10).join(f'- {feature}' for feature in app_info.key_features)}

**Ideal for:**
{chr(10).join(f'- {use_case}' for use_case in app_info.ideal_for)}

**Documentation:** {app_info.documentation_url}
    """

def get_model_info(model_name: str) -> str:
    """Return information about AI models.

    Args:
        model_name: The name of the model

    Returns:
        str: Formatted information about the model
    """
    info_models = {
        "gemini_pro_20": ModelInfo(
            name="Gemini Pro 2.0",
            provider="Google",
            description="Google's powerful large language model with advanced code generation capabilities",
            strengths=[
                "High-quality code generation",
                "Follows instructions well",
                "Handles complex prompts",
                "Good documentation in generated code"
            ],
            limitations=[
                "Requires API key",
                "Rate limits may apply",
                "Can be relatively slower than local models"
            ],
            requires_api_key=True,
            is_available=True
        ),
        "codet5": ModelInfo(
            name="CodeT5-small",
            provider="Salesforce",
            description="Specialized code generation model fine-tuned specifically for programming tasks",
            strengths=[
                "Focused on code generation",
                "More lightweight than larger models",
                "Faster inference times",
                "Can run locally"
            ],
            limitations=[
                "Smaller context window",
                "Less general knowledge",
                "May produce simpler code"
            ],
            requires_api_key=False,
            is_available=True
        ),
        "t0_3b": ModelInfo(
            name="T0_3B",
            provider="BigScience",
            description="3 billion parameter language model trained on diverse datasets with zero-shot capabilities",
            strengths=[
                "General-purpose capabilities",
                "Good instruction following",
                "Diverse training data",
                "Balance of size and performance"
            ],
            limitations=[
                "Not specialized for code generation",
                "May require adaptation of templates",
                "Medium-sized model (3B parameters)"
            ],
            requires_api_key=False,
            is_available=True
        )
    }

    model_info = info_models.get(model_name.lower())
    if not model_info:
        return "No information available for this model."

    # Format the information
    api_key_note = "⚠️ **Requires API Key**" if model_info.requires_api_key else "✅ **No API Key Required**"

    return f"""
### {model_info.name}

**Provider:** {model_info.provider}

**Description:** {model_info.description}

**Strengths:**
{chr(10).join(f'- {strength}' for strength in model_info.strengths)}

**Limitations:**
{chr(10).join(f'- {limitation}' for limitation in model_info.limitations)}

{api_key_note}
    """