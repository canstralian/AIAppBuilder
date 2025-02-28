import streamlit as st
from models import generate_with_gemini, generate_with_codet5, generate_with_t0
from app_templates import get_streamlit_template, get_gradio_template
from utils import format_code, validate_code, export_code, get_app_type_info, get_model_info
import random

# Set page configuration
st.set_page_config(
    page_title="AI App Generator",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
with open('assets/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Application state
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'app_type' not in st.session_state:
    st.session_state.app_type = "streamlit"
if 'template_name' not in st.session_state:
    st.session_state.template_name = "blank"
if 'model_name' not in st.session_state:
    st.session_state.model_name = "gemini"
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

# Header
st.title("🧙‍♂️ AI Application Generator")
st.markdown("Generate AI-powered Streamlit and Gradio applications using multiple AI models")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # App type selection
    app_type = st.radio(
        "Application Type",
        ["Streamlit", "Gradio"],
        key="app_type_radio",
        index=0,
        on_change=lambda: setattr(st.session_state, 'app_type', st.session_state.app_type_radio.lower())
    )
    
    # Template selection
    if app_type == "Streamlit":
        template_options = ["blank", "data_viz", "file_upload", "form", "nlp", "image_classifier"]
    else:  # Gradio
        template_options = ["blank", "image_classifier", "text_gen", "audio", "chat"]
    
    template_name = st.selectbox(
        "Starting Template",
        template_options,
        key="template_name_select",
        index=0,
        on_change=lambda: setattr(st.session_state, 'template_name', st.session_state.template_name_select)
    )
    
    # Model selection
    model_name = st.radio(
        "AI Model",
        ["Gemini Pro 2.0", "CodeT5", "T0_3B"],
        key="model_name_radio",
        index=0,
        on_change=lambda: setattr(st.session_state, 'model_name', st.session_state.model_name_radio.lower().replace(" ", "_").replace(".", ""))
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
        model_info = get_model_info(model_name.lower().replace(" ", "_").replace(".", ""))
        st.markdown(model_info)
        
    # App templates preview
    with st.expander("Template Preview"):
        if app_type.lower() == "streamlit":
            st.code(get_streamlit_template(template_name)[:500] + "...", language="python")
        else:
            st.code(get_gradio_template(template_name)[:500] + "...", language="python")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Generate Application")
    
    # User input
    user_prompt = st.text_area(
        "Describe your application in detail",
        height=150,
        placeholder="Example: Create a data visualization app that allows users to upload a CSV file and visualize the data using different chart types like bar charts, line charts, and scatter plots."
    )
    
    # Examples accordion
    with st.expander("Need inspiration? Try these examples"):
        example_prompts = [
            "A simple image classifier that can identify dogs, cats, and birds using a pre-trained model.",
            "A sentiment analysis app that analyzes the sentiment of user-entered text and provides a positive, negative, or neutral rating.",
            "A data dashboard that visualizes COVID-19 statistics with interactive maps and charts.",
            "A file converter app that allows users to upload images and convert them to different formats."
        ]
        
        for i, example in enumerate(example_prompts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                # Set the example as the prompt
                st.session_state.user_prompt = example
                st.experimental_rerun()
    
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
                        st.session_state.template_name
                    )
                elif model_chosen == "codet5":
                    st.session_state.generated_code = generate_with_codet5(
                        user_prompt, 
                        st.session_state.app_type, 
                        st.session_state.template_name
                    )
                elif model_chosen == "t0_3b":
                    st.session_state.generated_code = generate_with_t0(
                        user_prompt, 
                        st.session_state.app_type, 
                        st.session_state.template_name
                    )
                
                # Format the code if requested
                if st.session_state.format_code:
                    st.session_state.generated_code = format_code(st.session_state.generated_code)
                
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
        st.experimental_rerun()
    
    # Display prompt history
    if st.session_state.prompt_history:
        with st.expander("Prompt History"):
            for i, prompt in enumerate(st.session_state.prompt_history):
                st.text_area(f"Prompt {i+1}", value=prompt, height=100, disabled=True, key=f"history_{i}")

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
                            st.session_state.template_name
                        )
                    elif new_model == "codet5":
                        st.session_state.generated_code = generate_with_codet5(
                            st.session_state.prompt_history[-1], 
                            st.session_state.app_type, 
                            st.session_state.template_name
                        )
                    elif new_model == "t0_3b":
                        st.session_state.generated_code = generate_with_t0(
                            st.session_state.prompt_history[-1], 
                            st.session_state.app_type, 
                            st.session_state.template_name
                        )
                    
                    # Format the code if requested
                    if st.session_state.format_code:
                        st.session_state.generated_code = format_code(st.session_state.generated_code)
                    
                    # Success message
                    st.success(f"Regenerated with {new_model.replace('_', ' ').title()} model")
                
                except Exception as e:
                    st.error(f"Error regenerating code: {str(e)}")
    else:
        st.info("Enter a prompt and click 'Generate App' to create your application code")
        
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
    unsafe_allow_html=True
)