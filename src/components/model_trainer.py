import sys
import os
from src.logger import logging
from src.exception import CustomException

from src.components import data_transformation
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from dataclasses import dataclass

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trained_model_file_config=ModelTrainerconfig()
        

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting dependant and independant variables from train and test dataset')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )

            models={
                    'LogisticRegression':LogisticRegression(),
                    'DecisionTreeClassifier':DecisionTreeClassifier(),
                    'RandomForest':RandomForestClassifier(),
                    'AdaBoost':AdaBoostClassifier(),
                    'GradientBoost':GradientBoostingClassifier(),
                    'KNN':KNeighborsClassifier(),
                    'SVMClassifier':SVC()
                }
            
            accuracy_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print('\n======================================================================\n')
            logging.info(f'Model accuracy report: {accuracy_report} ')

            ## getting the best score
            best_model_score=max(sorted(accuracy_report.values()))
            best_model_name=list(accuracy_report.keys())[list(accuracy_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]

            print(f'Best Model info ->, Model name: {best_model_name}, accuracy_score : {best_model_score}')
            print('\n========================================================================================\n')
            logging.info(f'Best Model info ->, Model name: {best_model_name}, accuracy_score : {best_model_score}')

            save_object(
                file_path=self.trained_model_file_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e,sys)