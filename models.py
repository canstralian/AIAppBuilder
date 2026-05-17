"""
AI model integration module for code generation.
Provides interfaces to multiple AI models including Gemini, CodeT5, and T0.
This module handles loading models, generating code, and providing fallbacks.
"""
from typing import Tuple, Optional, Any, Callable
import os
import re
import google.generativeai as genai
import app_templates

# Wrap transformer imports in try-except to handle PyTorch errors
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except RuntimeError as e:
    if "Tried to instantiate class" in str(e):
        print("Warning: PyTorch custom class issue detected. Using fallback mode.")
        # Create dummy classes to prevent errors
        class DummyAutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return None
        
        class DummyAutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return None
                
        AutoModelForSeq2SeqLM = DummyAutoModel
        AutoTokenizer = DummyAutoTokenizer
    else:
        # Re-raise if it's a different error
        raise


# Initialize Gemini
def initialize_gemini() -> bool:
    """Initialize the Gemini API with the API key from environment variables.

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            print(f"Error initializing Gemini: {str(e)}")
            return False
    return False


# Initial check
GEMINI_INITIALIZED = initialize_gemini()

# Model loading functions with caching to avoid reloading
_CODET5_MODEL = None
_CODET5_TOKENIZER = None
_T0_MODEL = None
_T0_TOKENIZER = None


def _load_model_with_error_handling(
    model_name: str, model_id: str
) -> Tuple[Optional[Any], Optional[Any]]:
    """Generic function to load a model and tokenizer with error handling.

    Args:
        model_name (str): Human-readable name of the model for logging
        model_id (str): Hugging Face model identifier

    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return model, tokenizer
    except (RuntimeError, ValueError, ImportError) as e:
        print(f"Error loading {model_name} model: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading {model_name} model: {str(e)}")
        return None, None


def get_codet5_model() -> Tuple[Optional[Any], Optional[Any]]:
    """Load CodeT5 model and tokenizer.

    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    global _CODET5_MODEL, _CODET5_TOKENIZER
    if _CODET5_MODEL is None:
        _CODET5_MODEL, _CODET5_TOKENIZER = _load_model_with_error_handling(
            "CodeT5", "Salesforce/codet5-small"
        )
    return _CODET5_MODEL, _CODET5_TOKENIZER


def get_t0_model() -> Tuple[Optional[Any], Optional[Any]]:
    """Load T0_3B model and tokenizer.

    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    global _T0_MODEL, _T0_TOKENIZER
    if _T0_MODEL is None:
        _T0_MODEL, _T0_TOKENIZER = _load_model_with_error_handling(
            "T0", "bigscience/T0_3B"
        )
    return _T0_MODEL, _T0_TOKENIZER


def generate_with_gemini(prompt: str, app_type: str, template_name: str) -> str:
    """Generate code using Gemini Pro model.

    Args:
        prompt (str): User's description of the desired application
        app_type (str): Type of application ('streamlit' or 'gradio')
        template_name (str): Name of the template to use as reference

    Returns:
        str: Generated code or fallback if generation fails
    """
    # Initialize Gemini if not already initialized
    if not initialize_gemini():
        return fallback_generation(
            app_type,
            template_name,
            prompt,
            "Gemini API key not set or invalid. Please add your API key in the sidebar.",
        )

    try:
        # Get template for context
        template = ""
        if app_type == "streamlit":
            template = app_templates.get_streamlit_template(template_name)
        else:
            template = app_templates.get_gradio_template(template_name)

        # Build the prompt with instructions
        prompt_with_context = f"""
        Generate a {app_type.capitalize()} application based on the following description:
        
        Description: {prompt}
        
        Requirements:
        1. The code should be complete and runnable
        2. Add appropriate comments to explain the code
        3. Follow best practices for {app_type} development
        4. Include error handling where appropriate
        5. Make the UI clean and user-friendly
        
        Here's a similar template for reference:
        ```python
        {template}
        ```
        
        Please provide only the complete Python code without explanations or markdown.
        """

        # Set up the Gemini model
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            model_name="gemini-pro", generation_config=generation_config
        )

        # Generate the response
        response = model.generate_content(prompt_with_context)

        # Extract and clean the code
        generated_code = response.text

        # Clean up any markdown code blocks if present
        if "```python" in generated_code:
            code_blocks = re.findall(r"```python\n(.*?)```", generated_code, re.DOTALL)
            if code_blocks:
                generated_code = code_blocks[0]

        return generated_code

    except RuntimeError as e:
        print(f"Runtime error in Gemini generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))
    except ValueError as e:
        print(f"Value error in Gemini generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))
    except Exception as e:
        print(f"Error in Gemini generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))


def generate_with_codet5(prompt: str, app_type: str, template_name: str) -> str:
    """Generate code using CodeT5-small model.

    Args:
        prompt (str): User's description of the desired application
        app_type (str): Type of application ('streamlit' or 'gradio')
        template_name (str): Name of the template to use as reference

    Returns:
        str: Generated code or fallback if generation fails
    """
    try:
        model, tokenizer = get_codet5_model()
        if model is None or tokenizer is None:
            return fallback_generation(
                app_type, template_name, prompt, "CodeT5 model could not be loaded"
            )

        # Get template for context
        template = ""
        if app_type == "streamlit":
            template = app_templates.get_streamlit_template(template_name)
        else:
            template = app_templates.get_gradio_template(template_name)

        # Build simplified prompt due to token limits
        simplified_prompt = f"Create a {app_type} app that: {prompt}"

        # Tokenize input
        inputs = tokenizer(
            simplified_prompt, return_tensors="pt", max_length=512, truncation=True
        )

        # Generate code
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.2,
        )

        # Decode the generated output
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # CodeT5 might generate incomplete code, so provide fallback
        if len(code.strip()) < 50 or "import" not in code:
            return adapt_template(template, prompt)

        return code

    except RuntimeError as e:
        print(f"Runtime error in CodeT5 generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))
    except ValueError as e:
        print(f"Value error in CodeT5 generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))
    except Exception as e:
        print(f"Error in CodeT5 generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))


def generate_with_t0(prompt: str, app_type: str, template_name: str) -> str:
    """Generate code using T0_3B model.

    Args:
        prompt (str): User's description of the desired application
        app_type (str): Type of application ('streamlit' or 'gradio')
        template_name (str): Name of the template to use as reference

    Returns:
        str: Generated code or fallback if generation fails
    """
    try:
        model, tokenizer = get_t0_model()
        if model is None or tokenizer is None:
            return fallback_generation(
                app_type, template_name, prompt, "T0 model could not be loaded"
            )

        # Get template since T0 is not specialized for code
        if app_type == "streamlit":
            template = app_templates.get_streamlit_template(template_name)
        else:
            template = app_templates.get_gradio_template(template_name)

        # T0 isn't specialized for code, so we'll adapt a template
        # based on the user's prompt
        return adapt_template(template, prompt)

    except RuntimeError as e:
        print(f"Runtime error in T0 generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))
    except ValueError as e:
        print(f"Value error in T0 generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))
    except Exception as e:
        print(f"Error in T0 generation: {str(e)}")
        return fallback_generation(app_type, template_name, prompt, str(e))


def fallback_generation(
    app_type: str, template_name: str, prompt: str, error_message: str
) -> str:
    """Generate fallback code when model generation fails.

    Args:
        app_type (str): Type of application ('streamlit' or 'gradio')
        template_name (str): Name of the template to use
        prompt (str): Original user prompt
        error_message (str): Error message to include in the comment

    Returns:
        str: Fallback code based on the template
    """
    if app_type == "streamlit":
        template = app_templates.get_streamlit_template(template_name)
    else:
        template = app_templates.get_gradio_template(template_name)

    return f"""# Error in code generation: {error_message}
# Using template as fallback

{template}

# TODO: Implement the following functionality based on the prompt:
# {prompt}
"""


def adapt_template(template: str, prompt: str) -> str:
    """Adapts a template based on the user's prompt.

    Args:
        template (str): Template code to adapt
        prompt (str): User's prompt to extract information from

    Returns:
        str: Adapted template code
    """
    # Basic adaptation - change comments and app title
    adapted_code = template

    # Replace app title if it exists in the template
    title_match = re.search(r'st\.title\(["\'](.+?)["\']\)', template)
    if title_match:
        original_title = title_match.group(1)
        new_title = generate_title_from_prompt(prompt)
        adapted_code = adapted_code.replace(
            f'st.title("{original_title}")', f'st.title("{new_title}")'
        )

    # Gradio title replacement
    title_match = re.search(r'title="(.+?)"', template)
    if title_match:
        original_title = title_match.group(1)
        new_title = generate_title_from_prompt(prompt)
        adapted_code = adapted_code.replace(
            f'title="{original_title}"', f'title="{new_title}"'
        )

    # Add a comment about the purpose based on the prompt
    header_comment = f"# App generated for: {prompt}\n"
    adapted_code = header_comment + adapted_code

    return adapted_code


def extract_keywords(prompt: str) -> list:
    """Extract potential keywords from the prompt.

    Args:
        prompt (str): User's prompt

    Returns:
        list: List of extracted keywords
    """
    common_words = {
        "a",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "app",
        "create",
        "build",
    }
    words = prompt.lower().replace(".", "").replace(",", "").split()
    return [word for word in words if word not in common_words and len(word) > 3]


def generate_title_from_prompt(prompt: str) -> str:
    """Generate a title from the user's prompt.

    Args:
        prompt (str): User's prompt

    Returns:
        str: Generated title
    """
    # Extract first sentence or use whole prompt if short
    first_sentence = prompt.split(".")[0]
    if len(first_sentence) > 50:
        words = first_sentence.split()
        title = " ".join(words[:5]) + "..."
    else:
        title = first_sentence

    # Capitalize first letter of each word
    title = " ".join(word.capitalize() for word in title.split())

    return title
