import ast
import base64

def format_code(code):
    """Format the generated code.

    Args:
        code (str): The code to format.

    Returns:
        str: The formatted code.
    """
    try:
        # Parse the code to validate syntax
        ast.parse(code)
        # Code is valid, no need for complex formatting at this stage
        return code
    except SyntaxError as e:
        # If there's a syntax error, attempt basic cleanup
        # For now, just return the code with a comment about the error
        return f"# Note: The generated code has a syntax error that may need fixing:\n# {str(e)}\n\n{code}"

def validate_code(code):
    """Validate if the code has proper syntax.

    Args:
        code (str): The code to validate.

    Returns:
        tuple: A tuple containing a boolean indicating whether the code is valid and an error message (if any).
    """
    try:
        ast.parse(code)
        return True, "Code syntax is valid."
    except SyntaxError as e:
        line_number = e.lineno
        error_message = str(e)
        return False, f"Syntax error at line {line_number}: {error_message}"
    except Exception as e:
        return False, f"Error validating code: {str(e)}"

def export_code(code, filename="app.py"):
    """Create a download link for the code file.

    Args:
        code (str): The code to export.
        filename (str, optional): The name of the file. Defaults to "app.py".

    Returns:
        str: HTML code for a download link.
    """
    b64 = base64.b64encode(code.encode()).decode()
    href = f'<a href="data:file/text;base64,{b64}" download="{filename}" class="download-btn">💾 Download {filename}</a>'
    return href

def get_app_type_info(app_type):
    """Return information about app types.

    Args:
        app_type (str): The type of app.

    Returns:
        str: Information about the app type.
    """
    info = {
        "streamlit": """
### Streamlit

**Description:** Streamlit is an open-source Python framework that makes it easy to create beautiful, custom web apps for machine learning and data science.

**Key Features:**
- Rapid prototyping
- Simple Python API
- Real-time updates
- Interactive widgets
- Easy deployment

**Ideal for:**
- Data visualization dashboards
- Machine learning demonstrations
- Simple web tools
- Data exploration apps
        """,
        "gradio": """
### Gradio

**Description:** Gradio is a Python library that allows you to quickly create customizable UI components for your machine learning models, APIs, and data processing pipelines.

**Key Features:**
- Easy interface creation
- Multiple input/output types
- Simplified deployment
- API generation
- Hugging Face integration

**Ideal for:**
- ML model demos
- Image/audio/text processing interfaces
- Multi-modal applications
- Interactive machine learning demos
        """
    }

    return info.get(app_type, "No information available for this app type.")

def get_model_info(model_name):
    """Return information about AI models.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: Information about the model.
    """
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

**Description:** Specialized code generation model fine-tuned specifically for programming tasks.

**Strengths:**
- Focused on code generation
- More lightweight than larger models
- Faster inference times
- Can run locally

**Limitations:**
- Smaller context window
- Less general knowledge
- May produce simpler code
        """,
        "t0_3b": """
### T0_3B

**Provider:** BigScience

**Description:** A 3 billion parameter language model trained on diverse datasets with zero-shot capabilities.

**Strengths:**
- General-purpose capabilities
- Good instruction following
- Diverse training data
- Balance of size and performance

**Limitations:**
- Not specialized for code generation
- May require adaptation of templates
- Medium-sized model (3B parameters)
        """
    }

    return info.get(model_name, "No information available for this model.")
