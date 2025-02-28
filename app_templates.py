def get_streamlit_template(template_name):
    """Returns template code for Streamlit apps"""
    
    templates = {
        "blank": """import streamlit as st

# Configure page
st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🚀",
    layout="wide"
)

# Main app
st.title("My Streamlit App")
st.write("Welcome to my app! Edit this template to create your own app.")

# Add your app components below
""",

        "data_viz": """import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Data Visualization App",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Data Visualization Dashboard")
st.write("Upload your CSV data or use the sample dataset to create visualizations")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Option to use sample data or upload
    data_option = st.radio(
        "Choose data source:",
        ["Use sample data", "Upload my data"]
    )
    
    # Visualization options
    chart_type = st.selectbox(
        "Select chart type:",
        ["Line Chart", "Bar Chart", "Histogram", "Scatter Plot", "Heatmap"]
    )

# Data loading
if data_option == "Use sample data":
    # Generate sample data
    data = pd.DataFrame({
        'x': range(1, 101),
        'y': np.random.randn(100).cumsum(),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    st.success("Using sample data")
else:
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
            # Use sample data as fallback
            data = pd.DataFrame({
                'x': range(1, 101),
                'y': np.random.randn(100).cumsum(),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
            st.info("Using sample data as fallback")
    else:
        st.info("Please upload a CSV file")
        # Use sample data as placeholder
        data = pd.DataFrame({
            'x': range(1, 101),
            'y': np.random.randn(100).cumsum(),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })

# Display data preview
st.subheader("Data Preview")
st.dataframe(data.head())

# Visualization
st.subheader("Visualization")
if 'data' in locals():
    # Select columns for visualization
    if len(data.columns) > 0:
        x_col = st.selectbox("Select X-axis column:", data.columns)
        
        if chart_type != "Histogram":
            y_col = st.selectbox("Select Y-axis column:", data.columns)
        
        # Create visualization based on selection
        if chart_type == "Line Chart":
            st.line_chart(data.set_index(x_col)[y_col])
            
        elif chart_type == "Bar Chart":
            st.bar_chart(data.set_index(x_col)[y_col])
            
        elif chart_type == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(data[x_col], bins=20)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
        elif chart_type == "Scatter Plot":
            fig, ax = plt.subplots()
            ax.scatter(data[x_col], data[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
            
        elif chart_type == "Heatmap":
            # Only show heatmap if we have numeric data
            numeric_cols = data.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) > 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.error("Not enough numeric columns for a heatmap")
    else:
        st.error("The dataset has no columns")
""",

        "file_upload": """import streamlit as st
import pandas as pd
import io
from PIL import Image
import base64

# Configure page
st.set_page_config(
    page_title="File Upload Demo",
    page_icon="📂",
    layout="wide"
)

# App title and description
st.title("File Upload and Processing Demo")
st.write("Upload different file types and see how they can be processed")

# Sidebar
with st.sidebar:
    st.header("Options")
    file_type = st.radio(
        "Select file type to upload:",
        ["CSV/Excel Data", "Image", "Text File"]
    )

# Main content
st.header(f"Upload a {file_type}")

if file_type == "CSV/Excel Data":
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Check file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"File '{uploaded_file.name}' loaded successfully!")
            
            # Display data info
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Data Statistics")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Basic data exploration
            if st.checkbox("Show column information"):
                st.write(df.dtypes)
            
            if st.checkbox("Show summary statistics"):
                st.write(df.describe())
            
            # Allow downloading processed data
            if st.button("Process Data (Remove NAs)"):
                processed_df = df.dropna()
                st.write(f"Processed data shape: {processed_df.shape}")
                st.dataframe(processed_df.head())
                
                # Create download link
                csv = processed_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error: {e}")

elif file_type == "Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image processing options
            st.subheader("Image Processing")
            processing_option = st.selectbox(
                "Select an image processing option:",
                ["Resize", "Rotate", "Grayscale", "None"]
            )
            
            if processing_option == "Resize":
                new_width = st.slider("New width:", 100, 1000, 300)
                ratio = new_width / image.width
                new_height = int(image.height * ratio)
                resized_image = image.resize((new_width, new_height))
                st.image(resized_image, caption="Resized Image", use_column_width=True)
                
            elif processing_option == "Rotate":
                rotation_angle = st.slider("Rotation angle:", 0, 360, 90)
                rotated_image = image.rotate(rotation_angle)
                st.image(rotated_image, caption="Rotated Image", use_column_width=True)
                
            elif processing_option == "Grayscale":
                grayscale_image = image.convert("L")
                st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)
            
        except Exception as e:
            st.error(f"Error: {e}")

elif file_type == "Text File":
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
    
    if uploaded_file is not None:
        try:
            # Read text file
            content = uploaded_file.read().decode("utf-8")
            
            # Display content
            st.subheader("File Content")
            st.text_area("Text content:", value=content, height=300)
            
            # Text analysis
            st.subheader("Text Analysis")
            text_stats = {
                "Characters": len(content),
                "Words": len(content.split()),
                "Lines": len(content.splitlines()),
            }
            
            st.write(text_stats)
            
            # Additional text processing options
            processing_option = st.selectbox(
                "Text processing options:",
                ["None", "Convert to Uppercase", "Convert to Lowercase", "Count Word Frequency"]
            )
            
            if processing_option == "Convert to Uppercase":
                st.text_area("Uppercase text:", value=content.upper(), height=200)
                
            elif processing_option == "Convert to Lowercase":
                st.text_area("Lowercase text:", value=content.lower(), height=200)
                
            elif processing_option == "Count Word Frequency":
                words = content.lower().split()
                word_freq = {}
                for word in words:
                    word = word.strip(".,!?:;-\"'()[]{}")
                    if word and len(word) > 1:  # Ignore single character words and empty strings
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Display as DataFrame
                word_freq_df = pd.DataFrame(
                    {"Word": list(word_freq.keys()), "Frequency": list(word_freq.values())}
                ).sort_values("Frequency", ascending=False)
                
                st.dataframe(word_freq_df)
                
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("This app demonstrates how to handle various file uploads in Streamlit")
""",

        "form": """import streamlit as st

# Configure page
st.set_page_config(
    page_title="Streamlit Form Demo",
    page_icon="📝",
    layout="wide"
)

# App title and description
st.title("Streamlit Form Demo")
st.write("This app demonstrates how to create and handle forms in Streamlit")

# Initialize session state variables
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'submissions' not in st.session_state:
    st.session_state.submissions = []

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Registration Form")
    
    # Create a form
    with st.form(key="registration_form"):
        st.write("Please fill out the form below")
        
        # Form fields
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        
        gender = st.selectbox(
            "Gender",
            ["Prefer not to say", "Male", "Female", "Non-binary", "Other"]
        )
        
        interests = st.multiselect(
            "Interests",
            ["Technology", "Sports", "Arts", "Science", "Music", "Travel", "Food", "Reading"]
        )
        
        experience = st.radio(
            "Programming Experience",
            ["Beginner", "Intermediate", "Advanced"]
        )
        
        bio = st.text_area("Short Bio", max_chars=500)
        
        terms = st.checkbox("I agree to the terms and conditions")
        
        # Submit button
        submitted = st.form_submit_button("Submit Form")
        
        if submitted:
            # Validate form
            if not name or not email:
                st.error("Please fill out all required fields")
            elif "@" not in email or "." not in email:
                st.error("Please enter a valid email address")
            elif not terms:
                st.error("You must agree to the terms and conditions")
            else:
                # Store form data
                form_data = {
                    "name": name,
                    "email": email,
                    "age": age,
                    "gender": gender,
                    "interests": interests,
                    "experience": experience,
                    "bio": bio
                }
                
                st.session_state.submissions.append(form_data)
                st.session_state.form_submitted = True
                st.success("Form submitted successfully!")

with col2:
    st.header("Submissions")
    
    if st.session_state.form_submitted:
        # Display the latest submission
        latest = st.session_state.submissions[-1]
        
        st.subheader("Latest Submission")
        st.write(f"**Name:** {latest['name']}")
        st.write(f"**Email:** {latest['email']}")
        st.write(f"**Age:** {latest['age']}")
        st.write(f"**Gender:** {latest['gender']}")
        st.write(f"**Experience:** {latest['experience']}")
        
        st.write("**Interests:**")
        if latest['interests']:
            for interest in latest['interests']:
                st.write(f"- {interest}")
        else:
            st.write("None selected")
            
        st.write("**Bio:**")
        st.write(latest['bio'] if latest['bio'] else "No bio provided")
        
        # Option to clear form
        if st.button("Clear Form"):
            st.session_state.form_submitted = False
    else:
        st.info("No submissions yet. Fill out the form to see results.")
    
    # Show all submissions
    if st.session_state.submissions:
        st.subheader("All Submissions")
        for i, submission in enumerate(st.session_state.submissions):
            with st.expander(f"Submission {i+1}: {submission['name']}"):
                for key, value in submission.items():
                    if key == "interests":
                        st.write(f"**{key.capitalize()}:** {', '.join(value) if value else 'None'}")
                    else:
                        st.write(f"**{key.capitalize()}:** {value}")

# Advanced form features
st.header("Advanced Form Features")

tab1, tab2 = st.tabs(["File Upload Form", "Conditional Form"])

with tab1:
    with st.form(key="file_upload_form"):
        st.write("Upload and process files")
        
        name = st.text_input("Document Name")
        file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])
        tags = st.text_input("Tags (comma separated)")
        
        submit_file = st.form_submit_button("Upload Document")
        
        if submit_file and file:
            st.success(f"Document '{name}' uploaded successfully!")
            st.write(f"Filename: {file.name}")
            st.write(f"Size: {file.size} bytes")
            st.write(f"Tags: {tags}")

with tab2:
    with st.form(key="conditional_form"):
        st.write("Form with conditional fields")
        
        form_type = st.selectbox(
            "Form Type",
            ["Personal", "Business", "Educational"]
        )
        
        # Basic fields for all form types
        name = st.text_input("Name or Organization")
        email = st.text_input("Contact Email")
        
        # Conditional fields based on form type
        if form_type == "Personal":
            st.write("Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100)
            hobby = st.text_input("Favorite Hobby")
            
        elif form_type == "Business":
            st.write("Business Information")
            company_size = st.selectbox(
                "Company Size",
                ["1-10", "11-50", "51-200", "201-500", "500+"]
            )
            industry = st.text_input("Industry")
            
        elif form_type == "Educational":
            st.write("Educational Information")
            institution = st.text_input("Institution Name")
            field = st.text_input("Field of Study")
            
        submit_conditional = st.form_submit_button("Submit")
        
        if submit_conditional:
            st.success("Form submitted successfully!")
            st.write(f"Form Type: {form_type}")
            st.write(f"Name/Organization: {name}")
            st.write(f"Email: {email}")
            
            if form_type == "Personal":
                st.write(f"Age: {age}")
                st.write(f"Hobby: {hobby}")
            elif form_type == "Business":
                st.write(f"Company Size: {company_size}")
                st.write(f"Industry: {industry}")
            elif form_type == "Educational":
                st.write(f"Institution: {institution}")
                st.write(f"Field of Study: {field}")
""",

        "nlp": """import streamlit as st
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="NLP Analysis App",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Text Analysis Tool")
st.write("Analyze text data with various NLP techniques")

# Sidebar for options
with st.sidebar:
    st.header("Options")
    
    # Text input option
    text_input_option = st.radio(
        "Text Input Method:",
        ["Sample Text", "Enter Your Text", "Upload Text File"]
    )
    
    # Analysis options
    st.subheader("Analysis Options")
    
    show_word_count = st.checkbox("Word Count", value=True)
    show_char_count = st.checkbox("Character Count", value=True)
    show_sentence_count = st.checkbox("Sentence Count", value=True)
    show_word_frequency = st.checkbox("Word Frequency", value=True)
    show_word_cloud = st.checkbox("Word Cloud", value=False)
    
    # Advanced options
    with st.expander("Advanced Options"):
        min_word_length = st.slider("Minimum word length for analysis:", 1, 10, 3)
        max_words_to_display = st.slider("Max words to display:", 10, 100, 25)
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
        custom_stopwords = st.text_input("Add custom stopwords (comma separated):")
        
        if custom_stopwords:
            custom_stopwords = [word.strip() for word in custom_stopwords.split(",")]
        else:
            custom_stopwords = []

# Main content
# Text input
if text_input_option == "Sample Text":
    sample_text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them."
    text = sample_text
    st.info("Using sample text about NLP. You can change this in the sidebar.")
    
elif text_input_option == "Enter Your Text":
    text = st.text_area("Enter your text for analysis:", height=200)
    
else:  # Upload file
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.success(f"File uploaded successfully!")
    else:
        text = ""
        st.info("Please upload a text file")

# Analyze text if provided
if text:
    # Display the text with expandable section if it's long
    if len(text) > 500:
        with st.expander("View Text"):
            st.write(text)
    else:
        st.subheader("Text to Analyze")
        st.write(text)
    
    # Text preprocessing
    processed_text = text
    
    # Convert to lowercase
    processed_text = processed_text.lower()
    
    # Remove punctuation
    processed_text = re.sub(r'[^\w\s]', '', processed_text)
    processed_text = re.sub(r'\d+', '', processed_text)
    
    # Split into words
    words = processed_text.split()
    
    # Filter by length and remove stopwords
    if remove_stopwords:
        # Common English stopwords
        stopwords = [
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "then", 
            "once", "here", "there", "when", "where", "why", "how", "all", "any", 
            "both", "each", "few", "more", "most", "other", "some", "such", "no", 
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
            "t", "can", "will", "just", "don", "should", "now"
        ]
        # Add custom stopwords
        stopwords.extend(custom_stopwords)
        
        # Filter words
        filtered_words = [word for word in words if word not in stopwords and len(word) >= min_word_length]
    else:
        filtered_words = [word for word in words if len(word) >= min_word_length]
    
    # Analysis results
    st.header("Analysis Results")
    
    # Show text stats
    col1, col2, col3 = st.columns(3)
    
    if show_char_count:
        with col1:
            st.metric("Character Count", len(text))
    
    if show_word_count:
        with col2:
            st.metric("Word Count", len(words))
    
    if show_sentence_count:
        with col3:
            # Simple sentence counting by splitting on periods, exclamation points, and question marks
            sentences = re.split(r'[.!?]+', text)
            # Remove empty strings
            sentences = [s for s in sentences if s.strip()]
            st.metric("Sentence Count", len(sentences))
    
    # Word frequency analysis
    if show_word_frequency and filtered_words:
        st.subheader("Word Frequency Analysis")
        
        # Calculate word frequencies
        word_freq = Counter(filtered_words)
        most_common = word_freq.most_common(max_words_to_display)
        
        # Display as bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        words_df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
        
        # Sort by frequency
        words_df = words_df.sort_values('Frequency')
        
        # Plot horizontal bar chart
        ax.barh(words_df['Word'], words_df['Frequency'], color='skyblue')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Word')
        ax.set_title(f'Top {len(most_common)} Words by Frequency')
        
        # Add frequency labels
        for i, v in enumerate(words_df['Frequency']):
            ax.text(v + 0.1, i, str(v), color='black', va='center')
        
        st.pyplot(fig)
        
        # Show table of words
        st.dataframe(words_df)
        
    # Word Cloud
    if show_word_cloud and filtered_words:
        try:
            from wordcloud import WordCloud
            
            st.subheader("Word Cloud")
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color="white", 
                max_words=150, 
                contour_width=3, 
                contour_color='steelblue'
            ).generate(' '.join(filtered_words))
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
        except ImportError:
            st.warning("WordCloud package is not installed. Cannot generate word cloud.")
            st.code("pip install wordcloud", language="bash")
else:
    st.info("Enter or upload some text to analyze")

# Footer
st.markdown("---")
st.caption("Text Analysis Tool powered by NLTK and Streamlit")
""",

        "image_classifier": """import streamlit as st
import numpy as np
from PIL import Image
import time

# Configure page
st.set_page_config(
    page_title="Image Classification Demo",
    page_icon="🖼️",
    layout="wide"
)

# App title and description
st.title("Image Classification Demo")
st.write("Upload an image to classify it using a pre-trained model")

# Sidebar
with st.sidebar:
    st.header("Model Settings")
    
    model_type = st.selectbox(
        "Select a model:",
        ["MobileNet", "ResNet50", "VGG16"]
    )
    
    confidence_threshold = st.slider(
        "Confidence threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    st.subheader("About")
    st.markdown("""
    This app demonstrates image classification using deep learning models.
    
    **Note:** This is a demo app, and it uses simulated predictions.
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    # Image upload
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Classify Image"):
            # Show a progress bar for the classification process
            with st.spinner("Classifying..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate processing time
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Simulate model prediction
                # In a real app, this would call an actual model
                if model_type == "MobileNet":
                    classes = ["cat", "dog", "bird", "car", "flower"]
                elif model_type == "ResNet50":
                    classes = ["person", "chair", "table", "building", "tree"]
                else:  # VGG16
                    classes = ["airplane", "boat", "truck", "bicycle", "motorcycle"]
                
                # Generate random probabilities
                probabilities = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
                
                # Sort by probability (descending)
                sorted_indices = np.argsort(probabilities)[::-1]
                sorted_classes = [classes[i] for i in sorted_indices]
                sorted_probs = [probabilities[i] for i in sorted_indices]
                
                # Filter by confidence threshold
                filtered_results = [(cls, prob) for cls, prob in zip(sorted_classes, sorted_probs) if prob >= confidence_threshold]
                
                # Success message
                st.success("Classification complete!")
                
                # Display results in the second column
                with col2:
                    st.header("Classification Results")
                    
                    if filtered_results:
                        # Show top prediction
                        st.subheader(f"Top prediction: {filtered_results[0][0].title()}")
                        
                        # Display all predictions above threshold
                        st.write("All predictions above threshold:")
                        
                        # Create a bar chart for the predictions
                        chart_data = {
                            "Class": [c.title() for c, _ in filtered_results],
                            "Probability": [float(p) for _, p in filtered_results]
                        }
                        
                        # Display as a bar chart
                        st.bar_chart(chart_data, x="Class", y="Probability")
                        
                        # Also show as a table
                        for i, (cls, prob) in enumerate(filtered_results):
                            st.write(f"{i+1}. **{cls.title()}**: {prob:.2f}")
                    else:
                        st.info(f"No predictions with confidence above {confidence_threshold}")
    else:
        st.info("Please upload an image file")
        
        # Sample images
        st.subheader("Or try a sample image:")
        
        sample_images = [
            {"name": "Sample Cat", "path": "https://storage.googleapis.com/gradio-static-files/cat.jpg"},
            {"name": "Sample Car", "path": "https://storage.googleapis.com/gradio-static-files/car.jpg"},
            {"name": "Sample Bird", "path": "https://storage.googleapis.com/gradio-static-files/bird.jpg"}
        ]
        
        # Display sample images as buttons
        cols = st.columns(len(sample_images))
        for i, sample in enumerate(sample_images):
            with cols[i]:
                st.image(sample["path"], caption=sample["name"], width=150)
                if st.button(f"Use {sample['name']}", key=f"sample_{i}"):
                    # In a real app, you would download and process the sample image
                    st.write(f"Selected {sample['name']} (This would trigger classification in a complete app)")

# Additional info
st.markdown("---")
with st.expander("How it works"):
    st.write("""
    This application uses a deep learning model to classify images. Here's how it works:
    
    1. You upload an image or select a sample image
    2. The image is preprocessed (resized, normalized)
    3. The preprocessed image is fed into a pre-trained neural network
    4. The model outputs probability scores for various classes
    5. We display the top predictions above your selected confidence threshold
    
    In a real application, this would use actual models like MobileNet, ResNet, or VGG
    that have been trained on datasets like ImageNet.
    """)
"""
    }
    
    return templates.get(template_name, templates["blank"])


def get_gradio_template(template_name):
    """Returns template code for Gradio apps"""
    
    templates = {
        "blank": """import gradio as gr

def hello_world(name):
    return f"Hello, {name}!"

# Create a simple interface
demo = gr.Interface(
    fn=hello_world,
    inputs=gr.Textbox(placeholder="Enter your name"),
    outputs="text",
    title="My Gradio App",
    description="A simple Gradio application"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",

        "image_classifier": """import gradio as gr
import numpy as np
from PIL import Image

# Placeholder function for image classification
def classify_image(image):
    # This would normally use a real ML model
    # For demo purposes, we'll return random class probabilities
    classes = ["cat", "dog", "bird", "fish", "other"]
    probabilities = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
    
    # Create a dictionary of class->probability
    results = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    
    # Sort by probability (descending)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_results

# Create the Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Image Classifier",
    description="Upload an image to classify it into categories",
    examples=[
        ["https://storage.googleapis.com/gradio-static-files/img_sample1.jpg"],
        ["https://storage.googleapis.com/gradio-static-files/img_sample2.jpg"],
    ],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",

        "text_gen": """import gradio as gr
import random

# Mock text generation function
def generate_text(prompt, max_length, temperature):
    # In a real app, this would call a language model
    # For demo, we'll generate placeholder text
    
    # Sample text snippets to concatenate
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "In the beginning, there was code.",
        "The future of AI remains to be written.",
        "Gradio makes it easy to create interactive demos."
    ]
    
    # Generate "AI" text by repeating samples
    output = prompt + " "
    current_length = len(output.split())
    
    while current_length < max_length:
        # Higher temperature = more randomness
        if random.random() < temperature:
            # Add a completely random sentence
            next_text = random.choice(sample_texts)
        else:
            # Add a more "coherent" follow-up
            if "AI" in prompt or "artificial intelligence" in prompt.lower():
                next_text = "The development of AI continues to accelerate at an unprecedented pace."
            elif "data" in prompt.lower():
                next_text = "Data analysis reveals patterns that might otherwise remain hidden."
            elif "future" in prompt.lower():
                next_text = "The future remains uncertain but full of possibilities."
            else:
                next_text = random.choice(sample_texts)
                
        output += " " + next_text
        current_length = len(output.split())
    
    return output

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(placeholder="Enter a prompt to start generating text", label="Prompt"),
        gr.Slider(minimum=10, maximum=100, value=30, step=5, label="Max Length (words)"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation Demo",
    description="Generate text based on a prompt with adjustable parameters",
    examples=[
        ["The future of AI is", 50, 0.7],
        ["Once upon a time in a land far away", 75, 0.9],
        ["Data analysis shows that", 40, 0.5]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",

        "audio": """import gradio as gr
import numpy as np
import random
import string

# Mock transcription function (would normally use a speech recognition model)
def transcribe_audio(audio):
    # In a real app, this would use a speech recognition model
    # For this demo, we'll return a placeholder transcription
    
    if audio is None:
        return "No audio detected. Please record or upload audio."
    
    # Get audio duration in seconds
    duration = len(audio[1]) / audio[0]
    
    # Generate random "transcription" based on audio length
    word_count = int(duration * 2)  # Assume 2 words per second
    
    # List of common words for more realistic output
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
    ]
    
    # Generate sentence-like structure
    words = []
    
    # Add some sentences
    sentences = max(1, int(word_count / 8))
    for i in range(sentences):
        sentence_length = random.randint(4, 10)
        
        # Start with capital letter
        sentence = [random.choice(common_words).capitalize()]
        
        # Add more words
        sentence.extend(random.choices(common_words, k=sentence_length-1))
        
        # Join and add period
        sentence_str = " ".join(sentence) + "."
        words.append(sentence_str)
    
    transcription = " ".join(words)
    
    return transcription

# Create the Gradio interface
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(source="microphone", type="numpy"),
    outputs=gr.Textbox(label="Transcription"),
    title="Audio Transcription Demo",
    description="Record or upload audio to transcribe it into text",
    examples=[
        ["https://storage.googleapis.com/gradio-static-files/sample_audio1.wav"],
        ["https://storage.googleapis.com/gradio-static-files/sample_audio2.wav"]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
""",

        "chat": """import gradio as gr
import random
import time

# Sample responses for different topics
responses = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What can I assist you with?",
        "Greetings! How may I be of service?"
    ],
    "weather": [
        "The weather forecast indicates mild temperatures with a chance of rain.",
        "It's looking sunny with clear skies today!",
        "Expect cloudy conditions with occasional showers."
    ],
    "help": [
        "I can assist with various queries. Just ask me anything!",
        "I'm here to help! What information are you looking for?",
        "You can ask me about various topics, and I'll do my best to assist."
    ],
    "default": [
        "That's an interesting question. Let me think about it.",
        "I understand what you're asking. Here's what I can tell you...",
        "Thank you for your question. From my knowledge..."
    ]
}

# Chat function
def chatbot(message, history):
    # Simulate thinking time
    time.sleep(0.5)
    
    # Convert message to lowercase for easier matching
    message_lower = message.lower()
    
    # Check for greetings
    if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
        response = random.choice(responses["greeting"])
    
    # Check for weather queries
    elif any(word in message_lower for word in ["weather", "forecast", "temperature", "rain"]):
        response = random.choice(responses["weather"])
    
    # Check for help requests
    elif any(word in message_lower for word in ["help", "assist", "support"]):
        response = random.choice(responses["help"])
    
    # Default response for other queries
    else:
        response = random.choice(responses["default"])
        
        # Add a more specific comment based on keywords
        if "how" in message_lower:
            response += " The process typically involves several steps..."
        elif "why" in message_lower:
            response += " There are multiple factors that contribute to this..."
        elif "when" in message_lower:
            response += " The timing depends on various circumstances..."
        elif "where" in message_lower:
            response += " The location varies depending on context..."
    
    return response

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=chatbot,
    title="Interactive Chat Demo",
    description="Ask any question or just have a conversation",
    examples=[
        "Hello there!",
        "What's the weather like today?",
        "Can you help me find information?",
        "How does this process work?",
        "Why is the sky blue?"
    ],
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
"""
    }
    
    return templates.get(template_name, templates["blank"])