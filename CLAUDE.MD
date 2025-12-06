# CLAUDE.MD - AI Application Generator

## Project Overview

This is an AI Application Generator built with Streamlit that enables users to automatically generate Streamlit and Gradio application code using multiple AI models (Google Gemini Pro, CodeT5, and T0_3B). The tool provides template selection, code validation, formatting, and export capabilities.

## Architecture

### Core Components

1. **main.py** - Main Streamlit application interface
   - UI layout and user interactions
   - Session state management
   - Theme toggle (light/dark mode)
   - Code generation orchestration

2. **models.py** - AI model implementations
   - Google Gemini Pro integration (using "gemini-pro" model via Google Generative AI API)
   - CodeT5 integration (Hugging Face Transformers)
   - T0_3B integration (Hugging Face Transformers)
   - Model initialization and caching

3. **app_templates.py** - Application templates
   - Streamlit templates (6 types)
   - Gradio templates (5 types)
   - Template retrieval logic

4. **utils.py** - Utility functions
   - Code validation
   - Code formatting
   - Helper functions

### Technology Stack

- **Frontend**: Streamlit
- **AI Models**:
  - Google Gemini Pro (API model: "gemini-pro", via google-generativeai)
  - CodeT5 (via transformers)
  - T0_3B (via transformers)
- **ML Framework**: PyTorch
- **Code Processing**: NLTK, AST
- **Data Processing**: Pandas
- **Visualization**: Matplotlib, Seaborn

## File Structure

```
AIAppBuilder/
├── main.py                 # Main Streamlit application
├── models.py               # AI model implementations
├── app_templates.py        # Template definitions
├── utils.py                # Utility functions
├── assets/
│   └── custom.css          # Custom CSS for styling
├── requirements.txt        # Python dependencies
├── README.md               # User-facing documentation
├── CLAUDE.MD               # This file - AI assistant guide
├── generated-icon.png      # Project icon
└── .github/
    └── workflows/
        └── pylint.yml      # GitHub Actions for linting
```

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose

### Streamlit Best Practices
- Use `st.cache_resource` for model loading
- Use `st.cache_data` for data processing
- Manage state with `st.session_state`
- Avoid blocking operations in main thread

### Model Integration
- Models are loaded lazily on first use
- Caching prevents repeated model loads
- Error handling for API failures
- Fallback mechanisms for model unavailability

## Common Tasks

### Running the Application
```bash
streamlit run main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
# Optional: Set Google Gemini API key
export GOOGLE_API_KEY=your_api_key_here
```

### Linting
```bash
pylint main.py models.py app_templates.py utils.py
```

## Key Features

### Template System
- **Streamlit Templates**: Blank, Data Visualization, File Upload, Interactive Form, NLP Analysis, Image Classification
- **Gradio Templates**: Blank, Image Classification, Text Generation, Audio Analysis, Chat Interface

### AI Model Selection
Users can choose from three different models, each with different characteristics:
- **Google Gemini Pro** (API model: "gemini-pro", displayed as "Gemini Pro 2.0" in UI): Most capable, requires API key
- **CodeT5**: Specialized for code generation
- **T0_3B**: General-purpose text-to-text model

### Code Processing Pipeline
1. User inputs description + selects template + selects model
2. Template context is combined with user prompt
3. AI model generates code
4. Code validation (optional)
5. Code formatting (optional)
6. Display and export options

## Testing Strategy

### Manual Testing
- Test each template type
- Test each AI model
- Test code validation
- Test theme toggle
- Test export functionality (download/copy)

### Quality Checks
- GitHub Actions runs Pylint on push/PR
- Code validation catches syntax errors
- Manual review of generated code quality

## State Management

The app uses `st.session_state` for:
- `theme`: Current theme ('light' or 'dark')
- `generated_code`: Generated code storage
- `app_type`: Application type selection
- `model_name`: Model selection
- `template_name`: Template selection
- `user_prompt`: User's text input
- `prompt_history`: History of prompts
- `theme_changed`: Flag to manage theme updates

## API Keys and Configuration

- **GOOGLE_API_KEY**: Required for Google Gemini Pro (gemini-pro model)
  - Can be set as environment variable
  - Can be entered in UI sidebar
  - Falls back to manual input if not set

## Deployment Considerations

### Replit Configuration
- Exposed on port 3000
- Streamlit app configured for external access
- `.replit` file contains run configuration

### Resource Requirements
- Models require significant memory (especially CodeT5 and T0_3B)
- PyTorch dependencies are large
- Consider GPU for faster model inference

## Troubleshooting

### Common Issues

1. **PyTorch Installation Issues**
   - Ensure compatible Python version (3.8+)
   - Check CUDA compatibility if using GPU

2. **Model Loading Failures**
   - Check internet connectivity
   - Verify HuggingFace model availability
   - Check available memory

3. **Port Conflicts**
   - Default Streamlit port is 8501
   - Replit configuration uses port 3000

4. **API Key Issues**
   - Verify GOOGLE_API_KEY is set correctly
   - Check API quota limits

## Contributing

Please refer to the project's GitHub repository for contribution guidelines.
## Recent Changes

- Added GitHub Actions workflow for Pylint
- Fixed theme toggle functionality
- Added emojis to README for better engagement
- Fixed PyTorch class errors
- Updated Replit configuration for port 3000

## Future Enhancements

Potential areas for improvement:
- Add more AI models (GPT-4, Claude, etc.)
- Implement code execution sandbox
- Add version control for generated code
- Support for more frameworks (Flask, FastAPI, etc.)
- Collaborative features
- Code comparison between models
- Performance metrics for generated code

## License

MIT License - See LICENSE file for details
