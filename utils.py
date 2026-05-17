from typing import Tuple
import streamlit as st
import ast
import base64


def format_code(code: str) -> str:
    """Format the generated code.

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
    if not code or not isinstance(code, str):
        return False, "Code must be a non-empty string"
    try:
        ast.parse(code)
        return True, "Code syntax is valid."
    except SyntaxError as e:
        line_number = e.lineno if e.lineno else 0
        error_message = str(e)
        return False, f"Syntax error at line {line_number}: {error_message}"
    except Exception as e:
        return False, f"Error validating code: {str(e)}"

def export_code(code: str, filename: str = "app.py") -> str:
    """Create a download link for the code file.

    Args:
        code: The code to validate

    Returns:
        CodeValidationResult: Detailed validation result with suggestions
    """
    if not code:
        return ""
    b64 = base64.b64encode(code.encode()).decode()
    href = f'<a href="data:file/text;base64,{b64}" download="{filename}" class="download-btn">💾 Download {filename}</a>'
    return href

def get_app_type_info(app_type: str) -> str:
    """Return information about app types.

    Args:
        code: The code to export
        filename: The name of the file (defaults to "app.py")

    Returns:
        str: HTML code for a download link

    Raises:
        ValueError: If the code is empty or filename is invalid
    """
    if not isinstance(app_type, str):
        return "Invalid app type"

    app_type = app_type.lower()
    info = {
        "streamlit": """
### Streamlit

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
    if not isinstance(model_name, str):
        return "Invalid model name"

    model_name = model_name.lower()
    info = {
        "gemini_pro_20": """
### Gemini Pro 2.0

**Provider:** Google

**Description:** Google's powerful large language model with advanced code generation capabilities.

**Strengths:**
- High-quality code generation
- Follows instructions well
- Handles complex prompts
- Good documentation in generated code

**Limitations:**
- May require API key
- Rate limits may apply
- Can be relatively slower than local models
        """,
        "codet5": """
### CodeT5-small

**Provider:** Salesforce

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