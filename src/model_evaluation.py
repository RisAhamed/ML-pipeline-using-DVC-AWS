import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from pathlib import Path
# Ensure the "logs" directory exists
from dvclive import Live
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str)-> dict:
    try:
        with open(params_path,"r") as f:
            params = yaml.safe_load(f)
        logger.debug('Parameters loaded successfully')
        return params
    except Exception as e:
        logger.error('Failed to load parameters: %s', e)
        raise Exception(f'Failed to load parameters: {str(e)}')
    
def load_model(model_path: str):
    try: 
        with open(model_path,"rb") as f:
            model = pickle.load(f)
        logger.debug('Model loaded successfully')
        return model
    except Exception as e:
        logger.error('Failed to load model: %s', e)
        raise Exception(f'Failed to load model: {str(e)}')
    

def load_data(file_path: str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data Loader From %s",file_path)
        return df
    except Exception as e:
        logger.error('Failed to load data: %s', e)
        raise Exception(f'Failed to load data: {str(e)}')
    
def evaluate_model(model,x_test: np.ndarray,y_test: np.ndarray)-> dict:
    try:
        y_pred =model.predict(x_test)
        y_pred_data =model.predict_proba(x_test)[:,1]
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc_score = roc_auc_score(y_test,y_pred_data)
        logger.debug('Model evaluation completed successfully')
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc_score": auc_score
        }
    except Exception as e:
        logger.error('Failed to evaluate model: %s', e)
        raise Exception(f'Failed to evaluate model: {str(e)}')
    
def save_results(results: dict,results_path: str)->None:
    try:
        os.makedirs(Path(results_path).parent,exist_ok=True)
        with open(results_path,"w") as f:
            json.dump(results,f,indent=4)
        logger.debug('Results saved successfully')
    except Exception as e:
        logger.error('Failed to save results: %s', e)
        raise Exception(f'Failed to save results: {str(e)}')
    

def evaluation_pipeline(model_path: str,data_path: str,results_path: str)->None:
    try:
        params = load_params("params.yaml")
        
   
        model = load_model(model_path)
        data = load_data(data_path)
        x_test,y_test = data.iloc[: ,:-1].values,data.iloc[:,-1].values
        results = evaluate_model(model,x_test,y_test)
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', results['accuracy'])
            live.log_metric('precision', results['precision'])
            live.log_metric('recall', results['recall'])
            live.log_metric('auc_score', results['auc_score'])
        
            live.log_params(params)
        save_results(results,results_path)
        logger.info('Evaluation pipeline completed successfully')
    except Exception as e:
        logger.error('Failed to complete evaluation pipeline: %s', e)
        raise Exception(f'Failed to complete evaluation pipeline: {str(e)}')
    
if __name__ == "__main__":
    model_path = "./models/model.pkl"
    data_path = "./data/processed/test_tfidf.csv"
    results_path = "./results/results.json"
    evaluation_pipeline(model_path,data_path,results_path)
