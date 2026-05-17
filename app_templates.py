"""
Template code for different types of Streamlit and Gradio applications.
Provides ready-to-use templates for various application types.
"""


def get_streamlit_template(template_name: str) -> str:
    """Returns template code for Streamlit apps.

    Args:
        template_name (str): Name of the template to retrieve

    Returns:
        str: Python code for the requested template
    """
    templates = {
        "blank": """
import streamlit as st

st.title("My Streamlit App")

# Add your app content here
st.write("Hello, world!")
""",
        "data_viz": """
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Visualization App")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Show the data
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Data statistics
    st.subheader("Data Statistics")
    st.write(df.describe())
    
    # Select columns for visualization
    st.subheader("Visualization")
    
    # Choose visualization type
    viz_type = st.selectbox("Select Visualization Type", 
                           ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Heatmap"])
    
    # Select columns based on visualization type
    if viz_type in ["Bar Chart", "Line Chart", "Histogram"]:
        column = st.selectbox("Select a column", df.columns)
        
        if viz_type == "Bar Chart":
            fig, ax = plt.subplots()
            df[column].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
            
        elif viz_type == "Line Chart":
            fig, ax = plt.subplots()
            df[column].plot(kind='line', ax=ax)
            st.pyplot(fig)
            
        elif viz_type == "Histogram":
            fig, ax = plt.subplots()
            df[column].plot(kind='hist', ax=ax)
            st.pyplot(fig)
            
    elif viz_type == "Scatter Plot":
        x_column = st.selectbox("Select X column", df.columns)
        y_column = st.selectbox("Select Y column", df.columns)
        
        fig, ax = plt.subplots()
        df.plot.scatter(x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)
        
    elif viz_type == "Heatmap":
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.error("Need at least 2 numeric columns for a correlation heatmap")
""",
        "file_upload": """
import streamlit as st
import pandas as pd
import io

st.title("File Upload App")
st.write("Upload a file and process it")

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt', 'xlsx', 'json'])

if uploaded_file is not None:
    # Get file details
    file_details = {
        "Filename": uploaded_file.name,
        "File size": uploaded_file.size,
        "File type": uploaded_file.type
    }
    
    st.write("### File Details")
    st.json(file_details)
    
    # Determine file type and process accordingly
    if uploaded_file.type == "text/csv":
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Simple data analysis
        st.write("### Data Summary")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        if st.checkbox("Show column information"):
            st.write(df.dtypes)
        
        if st.checkbox("Show summary statistics"):
            st.write(df.describe())
            
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        # Read Excel
        df = pd.read_excel(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
    elif uploaded_file.type == "application/json":
        # Read JSON
        df = pd.read_json(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
    elif uploaded_file.type == "text/plain":
        # Read text file
        text_data = uploaded_file.read().decode("utf-8")
        st.write("### Text Content (first 500 characters)")
        st.text(text_data[:500] + "..." if len(text_data) > 500 else text_data)
        
        # Text analysis
        if st.checkbox("Show text analysis"):
            word_count = len(text_data.split())
            line_count = len(text_data.splitlines())
            char_count = len(text_data)
            
            st.write(f"Word count: {word_count}")
            st.write(f"Line count: {line_count}")
            st.write(f"Character count: {char_count}")
            
    # Download processed data
    if "df" in locals():
        st.write("### Download Processed Data")
        
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"processed_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )
""",
        "form": """
import streamlit as st
import pandas as pd

st.title("Form Application")
st.write("Fill out the form below to submit your information")

# Create a form
with st.form(key="my_form"):
    # Form fields
    name = st.text_input("Name")
    email = st.text_input("Email")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    
    # Date selection
    date = st.date_input("Select a date")
    
    # Category selection
    category = st.selectbox("Category", ["Option 1", "Option 2", "Option 3"])
    
    # Multiple selection
    options = st.multiselect("Select one or more options", 
                            ["Feature A", "Feature B", "Feature C", "Feature D"])
    
    # Slider
    rating = st.slider("Rate your experience", 0, 10, 5)
    
    # Text area
    comments = st.text_area("Additional comments")
    
    # Checkbox for terms
    terms_agree = st.checkbox("I agree to the terms and conditions")
    
    # Submit button
    submit_button = st.form_submit_button(label="Submit")

# Handle form submission
if submit_button:
    if not terms_agree:
        st.error("You must agree to the terms and conditions")
    elif not name or not email:
        st.error("Name and Email are required fields")
    else:
        # Create a dictionary with form data
        form_data = {
            "Name": name,
            "Email": email,
            "Age": age,
            "Date": date,
            "Category": category,
            "Selected Options": ", ".join(options),
            "Rating": rating,
            "Comments": comments
        }
        
        # Convert to DataFrame for display
        df = pd.DataFrame([form_data])
        
        st.success("Form submitted successfully!")
        
        # Display the submitted data
        st.subheader("Submitted Information")
        st.table(df)
        
        # Option to download the data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="form_submission.csv",
            mime="text/csv"
        )
""",
        "nlp": """
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_resources()

st.title("NLP Text Analysis Tool")
st.write("Enter text to analyze it with NLP techniques")

# Text input
text_input = st.text_area("Enter your text here:", height=200)

if text_input:
    # Tokenization
    st.subheader("Text Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Stats", "Word Frequency", "Sentiment", "Advanced"])
    
    with tab1:
        # Basic text statistics
        words = word_tokenize(text_input)
        sentences = sent_tokenize(text_input)
        
        # Remove punctuation
        words_no_punc = [word for word in words if word not in string.punctuation]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words_no_punc if word.lower() not in stop_words]
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Word Count", len(words_no_punc))
        
        with col2:
            st.metric("Sentence Count", len(sentences))
            
        with col3:
            st.metric("Unique Words", len(set(word.lower() for word in words_no_punc)))
        
        # Average word length
        avg_word_length = sum(len(word) for word in words_no_punc) / len(words_no_punc) if words_no_punc else 0
        
        # Average sentence length
        avg_sent_length = len(words_no_punc) / len(sentences) if sentences else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Word Length", f"{avg_word_length:.2f}")
        
        with col2:
            st.metric("Avg Sentence Length", f"{avg_sent_length:.2f}")
    
    with tab2:
        # Word frequency analysis
        word_freq = Counter(word.lower() for word in filtered_words)
        
        # Create DataFrame for visualization
        word_freq_df = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
        
        st.subheader("Top 20 Words")
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Word', data=word_freq_df, ax=ax)
        st.pyplot(fig)
        
        # Show frequency table
        st.dataframe(word_freq_df)
        
    with tab3:
        st.info("Sentiment analysis feature coming soon!")
        
    with tab4:
        # Advanced NLP options
        st.subheader("Advanced NLP Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show Stemming"):
                stemmer = PorterStemmer()
                stemmed_words = [stemmer.stem(word) for word in filtered_words[:20]]
                
                # Create comparison DataFrame
                stem_df = pd.DataFrame({
                    'Original Word': filtered_words[:20],
                    'Stemmed Word': stemmed_words
                })
                
                st.write("Stemming (first 20 words)")
                st.dataframe(stem_df)
        
        with col2:
            if st.button("Show Lemmatization"):
                lemmatizer = WordNetLemmatizer()
                lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words[:20]]
                
                # Create comparison DataFrame
                lemma_df = pd.DataFrame({
                    'Original Word': filtered_words[:20],
                    'Lemmatized Word': lemmatized_words
                })
                
                st.write("Lemmatization (first 20 words)")
                st.dataframe(lemma_df)
""",
        "image_classifier": """
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained model
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

model = load_model()

st.title("Image Classification App")
st.write("Upload an image to classify it using MobileNetV2 pre-trained on ImageNet")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Resize and preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Check if the image has 3 channels (RGB)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Preprocess the image
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        with st.spinner("Classifying..."):
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Display results
        st.subheader("Classification Results")
        
        # Create a table of results
        results_data = {
            "Class": [pred[1] for pred in decoded_predictions],
            "Description": [pred[1].replace('_', ' ').title() for pred in decoded_predictions],
            "Confidence": [f"{pred[2]*100:.2f}%" for pred in decoded_predictions]
        }
        
        results_df = pd.DataFrame(results_data)
        st.table(results_df)
        
        # Visualization of top prediction
        top_pred = decoded_predictions[0]
        st.subheader(f"Top Prediction: {top_pred[1].replace('_', ' ').title()}")
        
        # Show confidence bar for top prediction
        st.progress(float(top_pred[2]))
        
    else:
        st.error("Uploaded image must be RGB (3 channels). Please upload another image.")
""",
    }

    return templates.get(template_name, templates["blank"])


def get_gradio_template(template_name: str) -> str:
    """Returns template code for Gradio apps.

    Args:
        template_name (str): Name of the template to retrieve

    Returns:
        str: Python code for the requested template
    """
    templates = {
        "blank": """
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

# Create the Gradio interface
demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Enter your name"),
    outputs=gr.Textbox(label="Greeting"),
    title="Simple Greeting App",
    description="Enter your name to get a personalized greeting"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",
        "image_classifier": """
import gradio as gr
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# Load pre-trained model
model = resnet50(pretrained=True)
model.eval()

# ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import urllib.request
with urllib.request.urlopen(LABELS_URL) as f:
    LABELS = [line.decode("utf-8").strip() for line in f.readlines()]

# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(img):
    # Preprocess the image
    img_tensor = preprocess(Image.fromarray(img))
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Format results
    result = [(LABELS[idx], float(prob)) for prob, idx in zip(top5_prob, top5_idx)]
    
    return {label: prob for label, prob in result}

# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classifier",
    description="ResNet50 model trained on ImageNet dataset",
    examples=["example1.jpg", "example2.jpg"]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",
        "text_gen": """
import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_text(prompt, max_length=100, temperature=0.7):
    # Encode the input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.92,
        top_k=50
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation with GPT-2",
    description="Generate text using OpenAI's GPT-2 model"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",
        "audio": """
import gradio as gr
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
from PIL import Image

def process_audio(audio):
    # Extract the sample rate and audio data
    sample_rate, data = audio
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Generate visualizations
    results = {}
    
    # 1. Waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(data) / sample_rate, len(data)), data)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    results["waveform"] = Image.open(buf)
    plt.close()
    
    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    results["spectrogram"] = Image.open(buf)
    plt.close()
    
    # 3. MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    results["mfcc"] = Image.open(buf)
    plt.close()
    
    # Audio features
    features = {
        "Duration (s)": len(data) / sample_rate,
        "Sample Rate": sample_rate,
        "Max Amplitude": np.max(np.abs(data)),
        "Min Amplitude": np.min(np.abs(data)),
        "Mean": np.mean(data),
        "Standard Deviation": np.std(data)
    }
    
    return results["waveform"], results["spectrogram"], results["mfcc"], features

# Create the Gradio interface
demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(),
    outputs=[
        gr.Image(label="Waveform"),
        gr.Image(label="Spectrogram"),
        gr.Image(label="MFCC"),
        gr.JSON(label="Audio Features")
    ],
    title="Audio Analysis Tool",
    description="Upload an audio file to analyze its waveform, spectrogram, and MFCCs",
    examples=[["example1.wav"], ["example2.mp3"]]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",
        "chat": """
import gradio as gr
import random
import time

# Simple chatbot response function
def chatbot(message, history):
    # Simulate typing delay
    time.sleep(random.uniform(0.5, 1.5))
    
    # Simple response logic
    message = message.lower()
    
    if "hello" in message or "hi" in message:
        return "Hello there! How can I help you today?"
    
    elif "how are you" in message:
        return "I'm just a computer program, but I'm functioning well! How about you?"
    
    elif "bye" in message or "goodbye" in message:
        return "Goodbye! Have a great day!"
    
    elif "thank" in message:
        return "You're welcome! Is there anything else I can help with?"
    
    elif "help" in message:
        return "I can assist with answering questions, providing information, or just chatting. What would you like to know?"
    
    elif "what" in message and "your name" in message:
        return "I'm a friendly AI assistant created using Gradio. You can call me Chatbot!"
    
    elif "weather" in message:
        return "I don't have access to real-time weather data. You might want to check a weather service for that information!"
    
    elif "joke" in message:
        jokes = [
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why did the bicycle fall over? Because it was two-tired!",
            "How does a penguin build its house? Igloos it together!"
        ]
        return random.choice(jokes)
    
    else:
        responses = [
            "Interesting point. Could you tell me more about that?",
            "I'm not sure I fully understand. Could you explain differently?",
            "That's something I'd like to learn more about.",
            "Thanks for sharing that with me.",
            "Let me think about that for a moment..."
        ]
        return random.choice(responses)

# Create the Gradio chat interface
demo = gr.ChatInterface(
    fn=chatbot,
    title="Simple AI Chatbot",
    description="Ask me anything, and I'll do my best to respond!",
    examples=["Hello there!", "How are you today?", "Tell me a joke", "What's your name?"],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear chat"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",
    }

    return templates.get(template_name, templates["blank"])
