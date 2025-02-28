import streamlit as st
from models import generate_with_gemini, generate_with_codet5, generate_with_t0
import app_templates
from utils import export_code, format_code, validate_code
import streamlit.components.v1 as components
import re

# Set page config
st.set_page_config(
    page_title="AI App Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'app_type' not in st.session_state:
    st.session_state.app_type = "streamlit"
if 'model' not in st.session_state:
    st.session_state.model = "gemini"
if 'template' not in st.session_state:
    st.session_state.template = "blank"
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
if 'edited_code' not in st.session_state:
    st.session_state.edited_code = ""
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = None

# Sidebar
with st.sidebar:
    st.title("AI App Generator")
    st.markdown("### Create Streamlit and Gradio apps with AI")
    
    # Model selection
    st.markdown("## Model Selection")
    model_option = st.radio(
        "Choose an AI model:",
        ["Gemini Pro 2.0", "CodeT5-small", "T0_3B"],
        index=0,
        help="Select which AI model to use for code generation"
    )
    
    if model_option == "Gemini Pro 2.0":
        st.session_state.model = "gemini"
    elif model_option == "CodeT5-small":
        st.session_state.model = "codet5"
    else:
        st.session_state.model = "t0"
    
    # App type selection
    st.markdown("## App Type")
    app_type = st.radio(
        "Select app framework:",
        ["Streamlit", "Gradio"],
        index=0,
        help="Choose which framework to generate code for"
    )
    st.session_state.app_type = app_type.lower()
    
    # Template selection
    st.markdown("## Templates")
    template_options = {
        "streamlit": [
            "Blank App", 
            "Data Visualization", 
            "File Uploader", 
            "Form Demo", 
            "NLP Analysis App"
        ],
        "gradio": [
            "Blank App", 
            "Image Classifier", 
            "Text Generation", 
            "Audio Transcription", 
            "Chat Interface"
        ]
    }
    
    selected_template = st.selectbox(
        "Choose a starting template:",
        template_options[st.session_state.app_type],
        index=0
    )
    
    template_map = {
        "Blank App": "blank",
        "Data Visualization": "data_viz",
        "File Uploader": "file_upload",
        "Form Demo": "form",
        "NLP Analysis App": "nlp",
        "Image Classifier": "image_classifier",
        "Text Generation": "text_gen",
        "Audio Transcription": "audio",
        "Chat Interface": "chat"
    }
    
    st.session_state.template = template_map[selected_template]
    
    # Info section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses AI to generate app code based on your description.
    
    **Models:**
    - **Gemini Pro 2.0**: Google's latest LLM
    - **CodeT5-small**: Specialized for code generation
    - **T0_3B**: General-purpose text-to-text model
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit v1.42.2")

# Main content
st.title("Generate Your App")

# Prompt input
st.markdown("## Describe Your App")
prompt_placeholder = "Describe what you want your app to do. For example: 'Create a streamlit app that allows users to upload an image and apply filters to it.'"
user_prompt = st.text_area("Enter your prompt:", value=st.session_state.prompt, height=150, placeholder=prompt_placeholder)
st.session_state.prompt = user_prompt

# Load template example
if st.session_state.template != "blank":
    with st.expander("View template example", expanded=False):
        if st.session_state.app_type == "streamlit":
            st.code(app_templates.get_streamlit_template(st.session_state.template), language="python")
        else:
            st.code(app_templates.get_gradio_template(st.session_state.template), language="python")

# Generate button
col1, col2 = st.columns([1, 4])
with col1:
    generate_clicked = st.button("🚀 Generate Code", type="primary", use_container_width=True)

# Generate code when button is clicked
if generate_clicked and st.session_state.prompt:
    with st.spinner(f"Generating code with {model_option}..."):
        try:
            if st.session_state.model == "gemini":
                generated_code = generate_with_gemini(
                    st.session_state.prompt, 
                    st.session_state.app_type,
                    st.session_state.template
                )
            elif st.session_state.model == "codet5":
                generated_code = generate_with_codet5(
                    st.session_state.prompt, 
                    st.session_state.app_type,
                    st.session_state.template
                )
            else:  # t0
                generated_code = generate_with_t0(
                    st.session_state.prompt, 
                    st.session_state.app_type,
                    st.session_state.template
                )
                
            # Extract Python code from markdown if model returned markdown
            if "```python" in generated_code:
                code_blocks = re.findall(r"```python\n(.*?)```", generated_code, re.DOTALL)
                if code_blocks:
                    generated_code = code_blocks[0]
                    
            formatted_code = format_code(generated_code)
            st.session_state.generated_code = formatted_code
            st.session_state.edited_code = formatted_code
            st.session_state.generation_status = "success"
        except Exception as e:
            st.error(f"Error generating code: {str(e)}")
            st.session_state.generation_status = "error"

# Display generated code and editor
if st.session_state.generated_code:
    st.markdown("## Generated Code")
    st.markdown("You can edit the code below before exporting:")
    
    # Create a stacked layout: code editor on top, preview at bottom
    code_col, preview_col = st.columns(2)
    
    with code_col:
        st.markdown("### Code Editor")
        edited_code = st.text_area(
            "Edit your code:",
            value=st.session_state.generated_code,
            height=400,
            key="code_editor"
        )
        st.session_state.edited_code = edited_code
        
        # Provide code validation
        if st.button("Validate Code"):
            is_valid, message = validate_code(st.session_state.edited_code)
            if is_valid:
                st.success("Code validation passed! No syntax errors detected.")
            else:
                st.error(f"Code validation failed: {message}")
        
        # Export options
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download Code", key="download_btn"):
                filename = "app.py"
                export_code(st.session_state.edited_code, filename)
        
        with col2:
            if st.button("Copy to Clipboard", key="copy_btn"):
                # Using a JavaScript hack to copy to clipboard
                text_to_copy = st.session_state.edited_code.replace("'", "\\'").replace("\n", "\\n")
                st.markdown(f"""
                <script>
                    navigator.clipboard.writeText('{text_to_copy}');
                    alert('Code copied to clipboard!');
                </script>
                """, unsafe_allow_html=True)
                st.success("Code copied to clipboard!")
    
    with preview_col:
        st.markdown("### App Preview")
        st.info("This is a simplified preview and may not represent all functionalities.")
        
        # Create a mock preview of the app
        if st.session_state.app_type == "streamlit":
            st.markdown("""
            <div class="app-preview streamlit-preview">
                <div class="preview-header">Streamlit App Preview</div>
                <div class="preview-content">
                    <div class="mock-title">Your Streamlit App</div>
                    <div class="mock-widget"></div>
                    <div class="mock-text">App content would appear here</div>
                    <div class="mock-button">Run</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:  # gradio
            st.markdown("""
            <div class="app-preview gradio-preview">
                <div class="preview-header">Gradio App Preview</div>
                <div class="preview-content">
                    <div class="mock-title">Your Gradio Interface</div>
                    <div class="mock-input-box">Input</div>
                    <div class="mock-arrow">➔</div>
                    <div class="mock-output-box">Output</div>
                    <div class="mock-button">Submit</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### App Dependencies")
        if st.session_state.app_type == "streamlit":
            st.code("streamlit==1.42.2", language="text")
        else:
            st.code("gradio\nnumpy\npillow", language="text")

# Show examples if no code generated yet
if not st.session_state.generated_code:
    st.markdown("## Examples")
    
    # Create tabs for different examples
    example_tab1, example_tab2 = st.tabs(["Streamlit Example", "Gradio Example"])
    
    with example_tab1:
        st.markdown("### Sample Streamlit Data Visualization App")
        st.code(app_templates.get_streamlit_template("data_viz"), language="python")
    
    with example_tab2:
        st.markdown("### Sample Gradio Image Classification App")
        st.code(app_templates.get_gradio_template("image_classifier"), language="python")

# Instructions section
st.markdown("---")
st.markdown("## How to Use")
st.markdown("""
1. **Select a model** from the sidebar
2. **Choose app type** (Streamlit or Gradio)
3. **Pick a template** or start from blank
4. **Describe your app** in the prompt field
5. Click **Generate Code**
6. **Edit** the generated code if needed
7. **Export** your finished application
""")
