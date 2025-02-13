import pandas as pd
from pandas import DataFrame as df
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str)-> dict:
    try:
         with open(params_path,"r") as file:
              params = yaml.safe_load(file)
              logger.info("Parameters loaded successfully")
              return params
    except Exception as e:
        logger.error(f"An error occurred while loading parameters: {str(e)}")
        raise Exception(f"An error occurred while loading parameters: {str(e)}")

def load_data(data_path: str)-> pd.DataFrame:
    try: 
          df =pd.read_csv(data_path)
          df.fillna("",inplace= True)
          logger.info("Data loaded successfully")
          return df
    except Exception as e:
        logger.error(f"An error occurred while loading data: {str(e)}")
        raise Exception(f"An error occurred while loading data: {str(e)}")
    
def tfidf(train_data: df, test_data: df, max_features: int) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


def save_data(data: df,file_path:str)->None:
     try:
          os.makedirs(os.path.dirname(file_path),exist_ok= True)
          data.to_csv(file_path,index= False)
          logger.info(f"Data saved successfully to {file_path}")
     except Exception as e:
          logger.error(f"An error occurred while saving data: {str(e)}")
          raise Exception(f"An error occurred while saving data: {str(e)}")


def main():
    try:    
            params = load_params("params.yaml")
            max_features = params['feature_engineering']['max_features']
            train_data = load_data('./data/interim/train_processed.csv')
            test_data = load_data('./data/interim/test_processed.csv')
    
            train_df, test_df = tfidf(train_data, test_data, max_features)
    
            save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
            save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
            logger.error('Failed to complete the feature engineering process: %s', e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
