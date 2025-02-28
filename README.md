# AI Application Generator

![AI Application Generator](generated-icon.png)

## Overview

The AI Application Generator is a tool that automates the creation of Streamlit and Gradio applications using multiple AI models. This tool helps developers, data scientists, and software engineers to quickly prototype and generate code for various applications based on simple text descriptions.

## Features

- **Multiple AI Models**: Powered by Gemini Pro 2.0, CodeT5, and T0_3B
- **Template Selection**: Choose from various templates for both Streamlit and Gradio
- **Light/Dark Mode**: Toggle between light and dark theme for comfortable coding
- **Code Validation**: Automatically validate generated code syntax
- **Code Formatting**: Format generated code for readability
- **Export Options**: Download generated code or copy to clipboard
- **Regeneration**: Easily regenerate code using different models

## Application Types

### Streamlit Apps
- Blank Template
- Data Visualization
- File Upload & Processing
- Interactive Form
- NLP Analysis App
- Image Classification

### Gradio Apps
- Blank Template
- Image Classification
- Text Generation
- Audio Analysis
- Chat Interface

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-app-generator.git
cd ai-app-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up your Google Gemini API key as an environment variable:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Select the application type (Streamlit or Gradio)
2. Choose a starting template
3. Select the AI model to use
4. Enter a detailed description of your desired application
5. Click "Generate App" to create your code
6. Download or copy the generated code to use in your projects

## Example Prompts

- "A simple image classifier that can identify dogs, cats, and birds using a pre-trained model."
- "A sentiment analysis app that analyzes the sentiment of user-entered text and provides a positive, negative, or neutral rating."
- "A data dashboard that visualizes COVID-19 statistics with interactive maps and charts."
- "A file converter app that allows users to upload images and convert them to different formats."

## Advanced Options

- Adjust temperature for model creativity
- Enable/disable code validation
- Enable/disable auto-formatting of code

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI (for Gemini API)
- Transformers (for CodeT5 and T0 models)
- PyTorch
- NLTK
- Pandas
- Matplotlib
- Seaborn

## Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand our community expectations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by Hugging Face Spaces like [Deepseek Coder](https://huggingface.co/spaces/deepseek-ai/deepseek-coder-33b-instruct), [CodeLlama Playground](https://huggingface.co/spaces/codellama/codellama-playground), and [AI Python Code Reviewer](https://huggingface.co/spaces/whackthejacker/ai-python-code-reviewer)
- Thanks to all contributors who have helped shape this project