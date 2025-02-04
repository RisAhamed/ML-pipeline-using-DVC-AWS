import os,logger
import pandas as pd
from src.data_ingestion import *
from src.data_preprocessing import *
from src.feature_engineering import *

if __name__ == "__main__":
    data= data_ingestion(data_url="https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv",save_dir="data")
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

    main()