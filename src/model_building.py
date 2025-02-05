import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml
from  pandas import DataFrame as df
from pathlib import Path
# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str)->dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded successfully')
        return params
    except Exception as e:
        logger.error('Failed to load parameters: %s', e)
        raise Exception(f'Failed to load parameters: {str(e)}')

def load_data(file_path: str)-> df:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded successfully')
        return df
    except Exception as e:
        logger.error('Failed to load data: %s', e)
        raise Exception(f'Failed to load data: {str(e)}')

def save_model(model,file_path: str)-> None:
    try:
        os.makedirs(Path(file_path).parent,exist_ok=True)
        with open(file_path,"wb") as f:
            pickle.dump(model,f)
        logger.debug('Model saved successfully')
    except Exception as e:
        logger.error('Failed to save model: %s', e)
        raise Exception(f'Failed to save model: {str(e)}')
    
def train_model(x_train:np.ndarray,y_train: np.ndarray,params: dict)-> RandomForestClassifier:
    try:
        if x_train.shape[0]!=y_train.shape[0]:
            raise ValueError('x_train and y_train must have the same number of samples')
        
        logger.debug('Training model with parameters: %s', params)
        model = RandomForestClassifier(**params)
        model.fit(x_train,y_train)
        logger.debug('Model trained successfully')
        return model
    except Exception as e:
        logger.error('Failed to train model: %s', e)
        raise Exception(f'Failed to train model: {str(e)}')
                    


def model_pipeline(data_path: str,model_path: str,params: dict)->None:
    try:
        data = load_data(data_path)
        x_train,y_train = data.iloc[: ,:-1].values,data.iloc[:,-1].values
        model = train_model(x_train,y_train,params)
        save_model(model,model_path)
        logger.info('Model pipeline completed successfully')
    except Exception as e:
        logger.error('Failed to complete the model pipeline: %s', e)
        raise Exception(f'Failed to complete the model pipeline: {str(e)}')


if __name__ =="__main__":
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }
    
    model_pipeline('./data/processed/train_tfidf.csv','./models/model.pkl',params)