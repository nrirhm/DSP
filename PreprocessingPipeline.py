import re
import pandas as pd
import malaya

class PreprocessingPipeline:
    def __init__(self):
        # Load Malaya model
        self.tokenizer = malaya.tokenizer.Tokenizer()

    def clean_text(self,text):
        if not isinstance(text,str):
            return ''
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|https\S+|www\S+', '', text)
        # REmove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # not normalizing to retain slang words
    def tokenize_text(self,text):
        if not isinstance(text,str):
            return ''
        # Tokenize text
        return self.tokenizer.tokenize(text)
    
    # Processing a single entry
    def preprocess_text(self,text):
        cleaned_text = self.clean_text(text)
        tokenized_cleaned_text = self.tokenize_text(text)

        # create data frame with both output
        result_df = pd.DataFrame({
            'cleaned_text' : [cleaned_text],
            'tokenized_cleaned_text': [tokenized_cleaned_text]
        })    
        return result_df