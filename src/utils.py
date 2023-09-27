import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        accuracy_report={}
        
        for i in range(len(models)):
            model=list(model.values())[i]
            ## train model
            model.fit(X_train,y_train)

            #predict the test data
            y_test_pred=model.predict(X_test)

            ## get the evaluation scores for the model
            test_model_score=accuracy_score(y_test,y_test_pred)
            accuracy_report[list(model.keys())[i]]=test_model_score

        return accuracy_report


    except Exception as e:
        raise CustomException(e,sys)