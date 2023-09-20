import os
import sys 
sys.path.append('F:\DATA SCIENCE\Projects\Adult Census project')
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import pandas as pd


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)