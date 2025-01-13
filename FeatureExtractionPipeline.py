import re
import os
import pandas as pd
import openai
from dotenv import load_dotenv

class FeatureExtractionPipeline:
    def __init__(self):
        # Load environment variables from api_key.env file
        load_dotenv('C:/Users/irham/OneDrive/Desktop/Python Program/DSP/myenv/api_key.env')  # Specify your custom .env filename here
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.initialize_openai_client(api_key)
        else:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        # Initialize slang, reliable user, news words, and category mapping
        self.slang_words = ['aq', 'aku', 'ko', 'kau', 'deme', 'diorang', 'dorang', 'krn', 'sbb', 'skrg', 'apo', 'mcm', 'tula',
               'tulah', 'jer', 'je', 'ja', 'pls', 'lgsg', 'haha', 'hahaha', 'x', 'tau', 'entah', 'ntah', 'pulak',
               'gi', 'gak', 'ga', 'dh', 'tp', 'td', 'blm', 'sbb', 'hrs', 'hmm', 'nya', 'klau', 'kalo', 'gila',
               'ok', 'xde', 'tkde', 'takde', 'takda', 'tk', 'takdak', 'bodo', 'kitorang', 'sorang',
               'pdhl', 'eh', 'kan', 'bapak', 'srs', 'sumpah', 'la', 'lah', 'hm', 'korang', 'bang', 'sis', 'yall',
               'yalls', 'geng', 'weh', 'sapa', 'sape', 'bro'
        ]
        self.reliable_username = ['BuletinTV3', 'MINDEFMalaysia', 'KemPendidikan', 'mynadma', 'ssmofficialpage', 'bharianmy', 'imigresenmy', 'JPenerangan',
                     'nresmalaysia', 'KBSMalaysia', 'SinarOnline', 'MyCheckMalaysiaOfficial', 'KUSKOPMalaysia', 'moworksmy', 'mysumber', '501Awani',
                     'beritartm','hmetromy', 'bernamadotcom', 'SinarOnline', 'MalaysiaGazette', 'AADKMalaysia', 'UMonline', 'Bernama_Radio',
                     'AtletMalaysia', 'JapenSarawak', 'APMtwitter', 'BernamaTV', 'JapenSarawak', 'PETRAMalaysia', 'beritaalhijrah', 'KDNPUTRAJAYA',
                     'tv2_rtm', 'ASTROARENA', 'MOTMalaysia', 'Bernama_Radio', 'jpmgov_', 'jpsmofficial', 'posmalaysia', 'JaPenWPKL', 'MyGCCMalaysia',
                     'kemdigital_gov', 'japenjohor', 'JaPenWPKL', 'KPDN_HQ', 'MOTMalaysia', 'japenselangor1', 'kpkt_gov', 'DOSM_BPAN', 'MyJAKIM',
                     'CIMBMalaysia', 'MyMOTAC', 'JapenPahang', 'KWSPMalaysia', 'StatsMalaysia', 'SPRMMalaysia', 'KKMPutrajaya', 'MYParlimen'
        ]  
        self.news_words = ['berita', 'tular', 'kelayakan', 'video', 'laporan', 'terkini', 'sumbangan', 'semak', 'jom', 'hebahan', 'semakan', 
                                    'daftar', 'amaran', 'maklumat', 'info', 'dijamin', 'deposit', 'telegram', 'lanjut', 'pautan', 'hubungi', 'transaksi', 
                                    'sahih', 'kejadian', 'bantuan', 'status', 'janji', 'ringgit', 'membantu', 'peluang', 'dakwa', 'rasmi'
        ]

        self.category_mapping = {'Casual': 0, 'Formal': 1, 'Friendly': 2, 'Spam': 3}
        self.client_initialized =True #Tracks openai client initialization

    def initialize_openai_client(self, api_key):
        """Initialize the OpenAI client"""
        openai.api_key = api_key
    
    # Function to classify tweets
    def classify_tweet(self, text):
        if not self.client_initialized:
            raise ValueError("OpenAI is not initialized. Call 'initialized_open_client' first.")
        try:
            # Use OpenAI API to classify the text
            prompt = f"""
            Here is a text. Classify the tone of this text into one of the following categories:
            - Casual (e.g., daily/emotional updates)
            - Formal (e.g., fact/news)
            - Friendly (e.g., offering something)
            - Spam (e.g., repetitive or meaningless text)

            Only return the category name exactly as it is (Casual, Formal, Friendly, or Spam) without any additional explanation, brackets, or examples.

            Text: {text}
            Category:
            """
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to classify Malay text based on tone and content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # More deterministic output
                max_tokens=10
            )
            classification = response.choices[0].message.content.strip()
            return classification

        except Exception as e:
            return "Error"  # Indicate an error
    
    # Encode category
    def encode_category(self, text):
        return self.category_mapping.get(text,0)
    
    # Function to detect slang words
    def contains_slang_words(self,text):
        if not isinstance(text, str):
            return 0
        return int(any(re.search(rf"\b{word}\b", text.lower()) for word in self.slang_words))
    
    # Function to detect news words
    def contains_news_words(self,text):
        if not isinstance(text, str):
            return 0
        return int(any(re.search(rf"\b{word}\b", text.lower()) for word in self.news_words))
    
    # Function to detect reliable tweet handles
    def contains_reliable_username(self,text):
        if not isinstance(text, str):
            return 0
        return int(any(re.search(rf"\b{word}\b", text, re.IGNORECASE) for word in self.reliable_username))  # usernames are unique
    
    # Process a single entry
    def process_text(self,text):
        return {
            'contains_slang_words': self.contains_slang_words(text),
            'contains_news_words': self.contains_news_words(text),
            'contains_reliable_username': self.contains_reliable_username(text),
            'category_encoded' :  self.encode_category(text)
        }
    
    # Dataframe
    def process_dataframe(self, df, text_column, username_column):
        df['contains_slang_words'] = df[text_column].apply(self.contains_slang_words),
        df['contains_news_words'] = df[text_column].apply(self.contains_news_words),
        df['contains_reliable_username'] = df[username_column].apply(self.contains_reliable_username),
        df['category_encoded'] = df[text_column].apply(self.encode_category)
        return df