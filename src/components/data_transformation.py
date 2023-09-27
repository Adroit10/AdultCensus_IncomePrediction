import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataTransormationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransormationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Data transformation initiated')
            ## categorical and numerical columns
            categorical_columns=['marital-status', 'sex', 'employment_info']
            numerical_columns=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss','hours-per-week']

            # defining the custom ranking for the columns
            marital_status_map=['unmarried','married']
            sex_map=['Female','Male']
            employement_map=['without_pay','Private','govt','self_employed']

            logging.info('Data transformation pipeline initiated')

            num_pipeline=Pipeline(

                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[marital_status_map,sex_map,employement_map])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ]
            )

            logging.info('Data transformation completed')

            return preprocessor
        


        except Exception as e:
            logging.info('Exception occures in Data transformation phase')
            raise CustomException(e,sys)
        
        
    def workclasses_transform(self,data):
        try:
            if data['workclass'] in ['Federal-gov','Local-gov','State-gov']:
                return 'govt'
            elif data['workclass'] in ['Self-emp-not-inc','Self-emp-inc']:
                return 'self_employed'
            elif data['workclass']=='Private':
                return 'Private'
            else:
                return 'without_pay'
        except Exception as e:
            raise CustomException(e,sys)
        
    def mar_status_transform(self,status):
        try:
            unmarried=['Never-married','Divorced','Separated',
            'Widowed']
            if status in unmarried:
                return 'unmarried'
            else :
                return 'married'

        except Exception as e:
            raise CustomException(e,sys)
            


    def initiate_data_transformation(self,train_data_path,test_data_path):
        
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            categorical_columns_trans=['workclass', 'marital-status','sex','salary']
            for feature in categorical_columns_trans:
                train_df[feature]=train_df[feature].str.replace(" ","")
                test_df[feature]=test_df[feature].str.replace(" ","")

            for col in ['workclass']:
                train_df[col].replace('?',np.nan)
                test_df[col].replace('?',np.nan)
            train_df['salary']=train_df['salary'].map({'<=50K':0,'>50K':1})
            test_df['salary']=test_df['salary'].map({'<=50K':0,'>50K':1})

            logging.info('Reading of train and test data is completed')
            logging.info(f'Train DataFrame head : \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n{test_df.head().to_string()}')

            logging.info('obtaining preprocessor object')
            preprocessing_obj=self.get_data_transformation_obj()
            target_column='salary'
            drop_columns=[target_column,'race','occupation','education','country','relationship']

            ## dividing the dataset into independant and dependant features
            ## for training data
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column]

            ## for test data
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column]

            ## Handling the multiple classes in the columns such as workclass and marital-status 
            input_feature_train_df['marital-status']=input_feature_train_df['marital-status'].apply(self.mar_status_transform)
            input_feature_test_df['marital-status']=input_feature_test_df['marital-status'].apply(self.mar_status_transform)

            input_feature_train_df['employment_info']=input_feature_train_df.apply(self.workclasses_transform,axis=1)
            input_feature_test_df['employment_info']=input_feature_test_df.apply(self.workclasses_transform,axis=1)

            logging.info('input_feature_test columns: {}'.format(input_feature_test_df.columns))
            logging.info('input_feature_train columns: {}'.format(input_feature_train_df.columns))
            logging.info(f'Training features DataFrame head : \n{input_feature_train_df.head().to_string()}')
            logging.info(f'target column format : \n{target_feature_test_df.head().to_string()}')

            input_feature_test_df=input_feature_test_df.drop(labels=['workclass'],axis=1)
            input_feature_train_df=input_feature_train_df.drop(labels=['workclass'],axis=1)

            logging.info("Data in the columns have been handled and the unnecessary columns hav ebeen dropped")

            ## Data transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprossing obj on training and test data')
            logging.info(f'Training features DataFrame head : \n{input_feature_train_df.head().to_string()}')

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:

            raise CustomException(e,sys)