import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logger.debug("Config loaded successfully")
    return config

def load_params(params_path: str)-> dict:
    try:
        with open(params_path,"r") as file:
            params = yaml.safe_load(file)
        logger.debug("Params loaded successfully")
        return params
    except FileNotFoundError:
        logger.error(f"File not found at {params_path}")
        raise FileNotFoundError(f"File not found at {params_path}")
    except Exception as e:
        logger.error(f"An error occurred while loading params: {str(e)}")
        raise Exception(f"An error occurred while loading params: {str(e)}")
    

def load_data(data_url: str)-> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.info(f"Data loaded successfully from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"An error occurred while loading data: {str(e)}")
        raise Exception(f"An error occurred while loading data: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {str(e)}")
        raise Exception(f"An unexpected error occurred while loading data: {str(e)}")
    
def preprocess_data(df: pd.DataFrame)-> pd.DataFrame:
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug("Data preprocessing completed successfully")
        return df

    except KeyError as e:
        logger.error(f"KeyError: {str(e)}")
        raise KeyError(f"KeyError: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while preprocessing data: {str(e)}")
        raise Exception(f"An unexpected error occurred while preprocessing data: {str(e)}")
    

def save_data(df_train: pd.DataFrame,df_test : pd.DataFrame, save_dir: str)->None:
    try :
        raw_data_path = os.path.join(save_dir,"raw data")
        os.makedirs(raw_data_path,exist_ok=True)
        df_train.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        df_test.to_csv(os.path.join(raw_data_path,"test.csv"),index =False)
        logger.info(f"Data saved successfully to {raw_data_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data: {str(e)}")
        raise Exception(f"An unexpected error occurred while saving data: {str(e)}")



def data_ingestion(data_url: str,save_dir: str,)->None:
    try:
        # config = load_config(params_path)
        # params = load_params(params_path)
        params = {'train_test_split': {'test_size': 0.2, 'random_state': 42}}
        df = load_data(data_url)
        df = preprocess_data(df)
        df_train, df_test = train_test_split(df, **params['train_test_split'])
        save_data(df_train, df_test, save_dir)
        logger.info("Data ingestion completed successfully")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data ingestion: {str(e)}")
        raise Exception(f"An unexpected error occurred during data ingestion: {str(e)}")


if __name__ =="__main__":
    data= data_ingestion(data_url="https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv",save_dir="data")