import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
import re
logs_dir= "logs"
os.makedirs(logs_dir,exist_ok=True)

logging.basicConfig(filename=os.path.join(logs_dir,"data_preprocessing.log"),level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def transform(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9]',' ',text)
        text = text.split()
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)
        return text
    except Exception as e:
        logger.error(f"An error occurred while transforming text: {str(e)}")
        raise Exception(f"An error occurred while transforming text: {str(e)}")

def preprocess_data(df: pd.DataFrame,text_col: str,target_col: str)-> pd.DataFrame:
    try:
        logger.info("Starting data preprocessing")
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        logger.debug("Target column encoded")

        df = df.drop_duplicates(keep="first")
        logger.debug("Duplicated Removed")

        df.loc[:,text_col]  = df[text_col].apply(transform)
        logger.debug("Text transformed")

        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"An error occurred while preprocessing data: {str(e)}")
        raise Exception(f"An error occurred while preprocessing data: {str(e)}")
    


