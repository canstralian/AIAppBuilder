import streamlit as st
import ast
import io
import base64

def format_code(code):
    """Format the generated code"""
    try:
        # Parse the code to validate syntax
        ast.parse(code)
        # Code is valid, no need for complex formatting at this stage
        return code
    except SyntaxError as e:
        # If there's a syntax error, attempt basic cleanup
        lines = code.split('\n')
        # Try to identify and fix common syntax issues
        # For now, just return the code with a comment about the error
        return f"# Note: The generated code has a syntax error that may need fixing:\n# {str(e)}\n\n{code}"

def validate_code(code):
    """Validate if the code has proper syntax"""
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
    """Create a download link for the code file"""
    buffer = io.BytesIO()
    buffer.write(code.encode())
    buffer.seek(0)
    
    # Create a download button
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="text/plain"
    )
    
    return True

def get_app_type_info(app_type):
    """Return information about app types"""
    info = {
        "streamlit": {
            "name": "Streamlit",
            "version": "1.42.2",
            "description": "Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.",
            "website": "https://streamlit.io/",
            "examples": ["Data dashboards", "Data visualization", "Machine learning apps", "Simple web tools"]
        },
        "gradio": {
            "name": "Gradio",
            "version": "Latest",
            "description": "Gradio is a Python library that allows you to quickly create customizable UI components for your machine learning models, APIs, and data processing pipelines.",
            "website": "https://gradio.app/",
            "examples": ["ML model demos", "Image processing tools", "NLP applications", "Audio processing"]
        }
    }
    
    return info.get(app_type, {})

def get_model_info(model_name):
    """Return information about AI models"""
    info = {
        "gemini": {
            "name": "Gemini Pro 2.0",
            "provider": "Google",
            "description": "Google's latest generative AI model with advanced text generation capabilities.",
            "strengths": ["General text generation", "Detailed code explanations", "Following complex instructions"],
            "limitations": ["May sometimes include explanations outside of code", "Can be verbose"]
        },
        "codet5": {
            "name": "CodeT5-small",
            "provider": "Salesforce",
            "description": "A smaller version of the CodeT5 model specifically fine-tuned for code generation tasks.",
            "strengths": ["Focused on code generation", "More concise output", "Understands code context"],
            "limitations": ["Smaller context window", "Less general knowledge than larger models"]
        },
        "t0": {
            "name": "T0_3B",
            "provider": "BigScience",
            "description": "A 3 billion parameter language model designed for zero-shot tasks.",
            "strengths": ["Good at following instructions", "Handles various prompts", "Efficient for smaller tasks"],
            "limitations": ["Not specialized for code", "Smaller than state-of-the-art models"]
        }
    }
    
    return info.get(model_name, {})
