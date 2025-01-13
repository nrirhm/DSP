import streamlit as st
from PreprocessingPipeline import PreprocessingPipeline
from FeatureExtractionPipeline import FeatureExtractionPipeline
import pickle
import pandas as pd
from PIL import Image

# Load trained model
MODEL_PATH = "C:/Users/irham/OneDrive/Desktop/UM/Y3/DSP/dsp irham/Logistic Regression.pkl"
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Initialize pipelines
preprocessor = PreprocessingPipeline()
feature_extractor = FeatureExtractionPipeline()

# Images and Emojis
social_media_img = Image.open("C:/Users/irham/OneDrive/Desktop/UM/Y3/DSP/X Logo.png")

# Streamlit Application
def main():
    st.set_page_config(page_title="Fake News Detection on Social Media", layout='centered')
    st.sidebar.title("Navigation 📍")
    
    # Set a custom theme
    st.markdown(
        """
        <style>
        body {
            background-color: #f3f4f6;
            color: #0f1111;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    pages = ["Overview 🌐", "News Classification 📰", "Feedback 📣"]
    choice = st.sidebar.radio("Go to", pages)

    if choice == "Overview 🌐":
        st.title("Project Overview")
        st.image(social_media_img, use_column_width=True)
        
        st.write("""
        ## Fake News Detection on Social Media
        This application aims to help users assess the reliability of tweets by classifying them as 
        **Non-News**, **Genuine**, or **Likely Fake News**.
        """)
        
        st.info("### Objectives")
        st.write(
            """
            - Provide a tool for users to check the credibility of social media posts.
            - Utilize AI and NLP techniques for text classification.
            """
        )

        st.success("### Target Stakeholders")
        st.write(
            """
            - **MCMC (Malaysian Communications and Multimedia Commission)**
            - **Journalists**
            - **General Public**
            """
        )

        st.warning("### AI Utilization")
        st.write(
            """
            - This app uses NLP preprocessing, feature extraction, and a pre-trained Logistic Regression model.
            """
        )

    elif choice == "News Classification 📰":
        st.title("Classify Tweet")
        
        # User Inputs with tooltips
        tweet_text = st.text_area("Enter the Tweet text:", help="Input the content of the tweet you want to classify.")
        username = st.text_input("Enter the Tweeting username:", help="Enter the username of the person who tweeted.")

        st.markdown("---")  # Horizontal line
        
        if st.button("Classify"):
            with st.spinner("Processing... Please wait."):
                try:
                    # Preprocess and extract basic features
                    preprocessed_df = preprocessor.preprocess_text(tweet_text)
                    preprocessed_text = preprocessed_df['cleaned_text'].values[0]
                    
                    # Classify tweet using OpenAI for text category
                    text_category = feature_extractor.classify_tweet(preprocessed_text)
                    
                    # Encode category
                    category_encoded = feature_extractor.encode_category(text_category)

                    # Additional feature extraction
                    contains_slang = feature_extractor.contains_slang_words(preprocessed_text)
                    contains_reliable_user = feature_extractor.contains_reliable_username(username)
                    
                    # Additional features based on preprocessed data
                    text_length = len(preprocessed_text)
                    word_count = len(preprocessed_text.split())

                    # Prepare the final feature set
                    features = pd.DataFrame({
                        'category_encoded': [category_encoded],
                        'contains_slang_words': [contains_slang],
                        'contains_reliable_username': [contains_reliable_user],
                        'text_length': [text_length],
                        'word_count': [word_count],
                    })

                    # Predict with the logistic model
                    prediction = model.predict(features.values)
                    labels = {0: "Non-News", 1: "Genuine", 2: "Likely Fake News"}
                    result = labels[prediction[0]]

                    if result == "Non-News":
                        st.subheader(f"Classification Result: 🤔 {result}")
                    elif result == "Genuine":
                        st.subheader(f"Classification Result: ✅ {result}")
                    elif result == "Likely Fake News":
                        st.subheader(f"Classification Result: 🚨 {result}")
                        st.warning("This might be Fake News. Please verify from official sources.")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    elif choice == "Feedback 📣":
        st.title("Provide Feedback")
        feedback = st.text_area("Please provide your feedback:")
        if st.button("Submit"):
            # Here you could save the feedback, e.g., to a database or file
            st.success("Thank you for your feedback! 😊")

if __name__ == "__main__":
    main()