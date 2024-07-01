from fastapi import FastAPI
import pickle

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,RobustScaler, OneHotEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.model_selection import KFold

app = FastAPI()

@app.get('/')
def root():
    return {'greeting': "hello"}


@app.get('/predict')
def predict(user_year, user_month, user_day, user_postcode, user_property_type, user_property_age, user_ground):
    userinput = {
        'year': [user_year],
        'month': [user_month],
        'day': [user_day],
        'postcode': [user_postcode],
        'property_type': [user_property_type],
        'property_age': [user_property_age],
        'ground': [user_ground]
    }
    with open('../models/preprocessor.pkl', 'rb') as processor_file:
        # preprocess
        preprocessor = pickle.load(processor_file)
        df_userinput = pd.DataFrame(userinput, index=[0])
        transformed_userinput = preprocessor.transform(df_userinput)
        print(transformed_userinput)
        # load model and predict
        # print('This is BEFORE we load the model')  # - code debuggers
        model = keras.models.load_model('../models/model.h5')
        # print('NOW is AFTER we load the model') # - code debuggers
        prediction_log_return = model.predict(transformed_userinput).flatten()[0]
        # print('END: flattened and returned the prediction_log_return: {prediction_log_return}') # - code debuggers
        return {'prediction':prediction_log_return}

# sample_data = {
#     'year': [2022],
#     'month': [6],
#     'day': [21],
#     'postcode': [N1 2JU],
#     'property_type': ['F'],
#     'property_age': ['N'],
#     'ground': ['F']
# }

if __name__ == '__main__':
    prediction = predict(user_year=2022, user_month=6, user_day=21, user_postcode='N1 2JU', user_property_type='F', user_property_age='N', user_ground='F')
    print(f'The prediction for the default values is: {prediction}')
