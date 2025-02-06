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
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_col] = df[text_col].apply(transform)
        logger.debug('Text column transformed')
        return df
    except Exception as e:
        logger.error(f"An error occurred while preprocessing data: {str(e)}")
        raise Exception(f"An error occurred while preprocessing data: {str(e)}")
    


if __name__=="__main__":
    train_data = pd.read_csv('./data/raw data/train.csv')
    test_data = pd.read_csv('./data/raw data/test.csv')
    logger.debug('Data loaded properly')

    # Transform the data
    train_processed_data = preprocess_data(train_data, "text","target")
    test_processed_data = preprocess_data(test_data, "text","target")

    # Store the data inside data/processed
    data_path = os.path.join("./data", "interim")
    os.makedirs(data_path, exist_ok=True)
    
    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    
    logger.debug('Processed data saved to %s', data_path)

