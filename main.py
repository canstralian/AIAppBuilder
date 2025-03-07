import streamlit as st
import os
from models import generate_with_gemini, generate_with_codet5, generate_with_t0
from app_templates import get_streamlit_template, get_gradio_template
from utils import (
    format_code,
    validate_code,
    export_code,
    get_app_type_info,
    get_model_info,
)
import random

# Set page configuration
st.set_page_config(
    page_title="AI App Generator",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize theme in session state if not present
if "theme" not in st.session_state:
    # Default to light theme
    st.session_state.theme = "light"

# Custom CSS
with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Check for API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
has_gemini_api_key = GOOGLE_API_KEY is not None

# Application state
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "app_type" not in st.session_state:
    st.session_state.app_type = "streamlit"
if "template_name" not in st.session_state:
    st.session_state.template_name = "blank"
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini"
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

# Header
st.title("🧙‍♂️ AI Application Generator")
st.markdown(
    "Generate AI-powered Streamlit and Gradio applications using multiple AI models"
)

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # Theme toggle
    if "theme_changed" not in st.session_state:
        st.session_state.theme_changed = False
        
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    theme_text = f"{theme_icon} Switch to {'Dark' if st.session_state.theme == 'light' else 'Light'} Mode"
    
    if st.button(theme_text):
        if st.session_state.theme == "light":
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"
        st.session_state.theme_changed = True
        st.rerun()

    # Apply theme-specific styles through HTML
    if st.session_state.theme == "dark":
        st.markdown(
            """
        <style>
        :root {
            --primary-color: #BB86FC;
            --secondary-color: #03DAC6;
            --background-color: #121212;
            --text-color: #E0E0E0;
            --accent-color: #808495;
            --light-accent-color: #2C2C35;
            --dark-accent-color: #424255;
            --success-color: #0CCE6B;
            --error-color: #CF6679;
            --warning-color: #FFC107;
            --info-color: #2196F3;
        }
        
        /* Dark theme overrides */
        .main {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .stTextInput, .stTextArea, .stSelectbox {
            background-color: #1E1E1E;
            color: white;
        }
        
        .stButton>button {
            color: white;
        }
        
        .stMarkdown {
            color: var(--text-color);
        }
        
        .css-1r6slb0, .css-1fv8s86 {  /* Sidebar background */
            background-color: #1E1E1E;
        }
        
        .stCodeBlock>div {
            background-color: #2D2D2D;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color);
        }
        
        .stExpander {
            border-color: #2D2D2D;
        }
        
        a {
            color: var(--secondary-color);
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    # API Key input (if needed)
    if not has_gemini_api_key:
        st.warning("⚠️ Gemini API Key not found")
        api_key = st.text_input(
            "Enter your Google Gemini API Key:",
            type="password",
            help="Get a key at https://makersuite.google.com/",
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("API Key set! Please refresh to apply.")
            if st.button("Refresh App"):
                st.rerun()
    else:
        st.success("✅ Gemini API Key detected")

    # App type selection
    app_type = st.radio(
        "Application Type",
        ["Streamlit", "Gradio"],
        key="app_type_radio",
        index=0,
        on_change=lambda: setattr(
            st.session_state, "app_type", st.session_state.app_type_radio.lower()
        ),
    )

    # Template selection
    template_display_names = {
        "streamlit": {
            "blank": "Blank Template",
            "data_viz": "Data Visualization",
            "file_upload": "File Upload & Processing",
            "form": "Interactive Form",
            "nlp": "NLP Analysis App",
            "image_classifier": "Image Classification",
        },
        "gradio": {
            "blank": "Blank Template",
            "image_classifier": "Image Classification",
            "text_gen": "Text Generation",
            "audio": "Audio Analysis",
            "chat": "Chat Interface",
        },
    }

    if app_type == "Streamlit":
        template_keys = [
            "blank",
            "data_viz",
            "file_upload",
            "form",
            "nlp",
            "image_classifier",
        ]
        template_options = [
            template_display_names["streamlit"][key] for key in template_keys
        ]
    else:  # Gradio
        template_keys = ["blank", "image_classifier", "text_gen", "audio", "chat"]
        template_options = [
            template_display_names["gradio"][key] for key in template_keys
        ]

    selected_display_name = st.selectbox(
        "Starting Template", template_options, key="template_display_select", index=0
    )

    # Find the corresponding template key
    if app_type == "Streamlit":
        selected_key = [
            k
            for k, v in template_display_names["streamlit"].items()
            if v == selected_display_name
        ][0]
    else:
        selected_key = [
            k
            for k, v in template_display_names["gradio"].items()
            if v == selected_display_name
        ][0]

    # Update session state
    st.session_state.template_name = selected_key

    # Model selection
    available_models = ["Gemini Pro 2.0", "CodeT5", "T0_3B"]
    if not has_gemini_api_key:
        # If no API key, show warning next to Gemini option
        available_models[0] = "Gemini Pro 2.0 ⚠️"

    model_name = st.radio(
        "AI Model",
        available_models,
        key="model_name_radio",
        index=1 if not has_gemini_api_key else 0,  # Default to CodeT5 if no API key
        on_change=lambda: setattr(
            st.session_state,
            "model_name",
            st.session_state.model_name_radio.split(" ")[0].lower().replace(".", "")
            + (
                st.session_state.model_name_radio.split(" ")[1].lower()
                if len(st.session_state.model_name_radio.split(" ")) > 1
                and st.session_state.model_name_radio.split(" ")[0].lower() == "gemini"
                else ""
            ),
        ),
    )

    # Advanced options
    with st.expander("Advanced Options"):
        st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature")
        st.checkbox("Enable code validation", value=True, key="validate_code")
        st.checkbox("Auto-format code", value=True, key="format_code")

    # Info sections
    with st.expander("About App Types"):
        app_type_info = get_app_type_info(app_type.lower())
        st.markdown(app_type_info)

    with st.expander("About AI Models"):
        model_info = get_model_info(
            model_name.lower().replace(" ", "_").replace(".", "")
        )
        st.markdown(model_info)

    # App templates preview
    with st.expander("Template Preview"):
        if app_type.lower() == "streamlit":
            st.code(
                get_streamlit_template(st.session_state.template_name)[:500] + "...",
                language="python",
            )
        else:
            st.code(
                get_gradio_template(st.session_state.template_name)[:500] + "...",
                language="python",
            )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Generate Application")

    # User input
    user_prompt = st.text_area(
        "Describe your application in detail",
        height=150,
        placeholder="Example: Create a data visualization app that allows users to upload a CSV file and visualize the data using different chart types like bar charts, line charts, and scatter plots.",
    )

    # Examples accordion
    with st.expander("Need inspiration? Try these examples"):
        example_prompts = [
            "A simple image classifier that can identify dogs, cats, and birds using a pre-trained model.",
            "A sentiment analysis app that analyzes the sentiment of user-entered text and provides a positive, negative, or neutral rating.",
            "A data dashboard that visualizes COVID-19 statistics with interactive maps and charts.",
            "A file converter app that allows users to upload images and convert them to different formats.",
        ]

        for i, example in enumerate(example_prompts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                # Set the example as the prompt
                st.session_state.user_prompt = example
                st.rerun()

    # Generate button
    generate_col1, generate_col2 = st.columns([1, 1])

    with generate_col1:
        generate_button = st.button("🪄 Generate App", use_container_width=True)

    with generate_col2:
        clear_button = st.button("🧹 Clear", use_container_width=True)

    # Handle generate button
    if generate_button and user_prompt:
        with st.spinner("Generating your application..."):
            # Log the prompt to history
            st.session_state.prompt_history.append(user_prompt)

            # Determine which model to use and generate code
            model_chosen = st.session_state.model_name

            try:
                if model_chosen == "gemini_pro_20":
                    st.session_state.generated_code = generate_with_gemini(
                        user_prompt,
                        st.session_state.app_type,
                        st.session_state.template_name,
                    )
                elif model_chosen == "codet5":
                    st.session_state.generated_code = generate_with_codet5(
                        user_prompt,
                        st.session_state.app_type,
                        st.session_state.template_name,
                    )
                elif model_chosen == "t0_3b":
                    st.session_state.generated_code = generate_with_t0(
                        user_prompt,
                        st.session_state.app_type,
                        st.session_state.template_name,
                    )

                # Format the code if requested
                if st.session_state.format_code:
                    st.session_state.generated_code = format_code(
                        st.session_state.generated_code
                    )

                # Validate the code if requested
                if st.session_state.validate_code:
                    is_valid, error_msg = validate_code(st.session_state.generated_code)
                    if not is_valid:
                        st.error(f"Generated code has syntax errors: {error_msg}")

                # Success message
                st.success("App generated successfully!")

            except Exception as e:
                st.error(f"Error generating code: {str(e)}")
                st.session_state.generated_code = "# Error occurred during generation"

    # Handle clear button
    if clear_button:
        st.session_state.generated_code = ""
        st.session_state.user_prompt = ""
        st.rerun()

    # Display prompt history
    if st.session_state.prompt_history:
        with st.expander("Prompt History"):
            for i, prompt in enumerate(st.session_state.prompt_history):
                st.text_area(
                    f"Prompt {i+1}",
                    value=prompt,
                    height=100,
                    disabled=True,
                    key=f"history_{i}",
                )

with col2:
    st.header("Code Preview")

    # Display the generated code with syntax highlighting
    if st.session_state.generated_code:
        st.code(st.session_state.generated_code, language="python")

        # Code actions
        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            download_link = export_code(st.session_state.generated_code)
            st.markdown(download_link, unsafe_allow_html=True)

        with action_col2:
            copy_button = st.button("📋 Copy to Clipboard", use_container_width=True)

        with action_col3:
            regenerate_button = st.button("🔄 Regenerate", use_container_width=True)

        # Regenerate with different model
        if regenerate_button:
            with st.spinner("Regenerating code..."):
                # Select a different model randomly
                current_model = st.session_state.model_name
                available_models = ["gemini_pro_20", "codet5", "t0_3b"]
                available_models.remove(current_model)
                new_model = random.choice(available_models)

                try:
                    if new_model == "gemini_pro_20":
                        st.session_state.generated_code = generate_with_gemini(
                            st.session_state.prompt_history[-1],
                            st.session_state.app_type,
                            st.session_state.template_name,
                        )
                    elif new_model == "codet5":
                        st.session_state.generated_code = generate_with_codet5(
                            st.session_state.prompt_history[-1],
                            st.session_state.app_type,
                            st.session_state.template_name,
                        )
                    elif new_model == "t0_3b":
                        st.session_state.generated_code = generate_with_t0(
                            st.session_state.prompt_history[-1],
                            st.session_state.app_type,
                            st.session_state.template_name,
                        )

                    # Format the code if requested
                    if st.session_state.format_code:
                        st.session_state.generated_code = format_code(
                            st.session_state.generated_code
                        )

                    # Success message
                    st.success(
                        f"Regenerated with {new_model.replace('_', ' ').title()} model"
                    )

                except Exception as e:
                    st.error(f"Error regenerating code: {str(e)}")
    else:
        st.info(
            "Enter a prompt and click 'Generate App' to create your application code"
        )

        # Preview of templates
        with st.expander("Preview Available Templates"):
            selected_template = st.session_state.template_name
            app_type = st.session_state.app_type

            if app_type == "streamlit":
                st.code(get_streamlit_template(selected_template), language="python")
            else:
                st.code(get_gradio_template(selected_template), language="python")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Built with ❤️ using Streamlit and multiple AI models</p>
        <p>Inspired by Hugging Face Spaces like <a href="https://huggingface.co/spaces/deepseek-ai/deepseek-coder-33b-instruct">Deepseek Coder</a>, 
        <a href="https://huggingface.co/spaces/codellama/codellama-playground">CodeLlama Playground</a>, and 
        <a href="https://huggingface.co/spaces/whackthejacker/ai-python-code-reviewer">AI Python Code Reviewer</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
