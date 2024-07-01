from fastapi import FastAPI, Request, HTTPException
from json import JSONDecodeError
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
def predict(year, month, day, postcode, property_type, property_age, ground):
    userinput = {
        'year': [year],
        'month': [month],
        'day': [day],
        'postcode': [postcode],
        'property_type': [property_type],
        'property_age': [property_age],
        'ground': [ground]
    }
    with open('../models/preprocessor.pkl', 'rb') as processor_file:
        # preprocess
        preprocessor = pickle.load(processor_file)
        df_userinput = pd.DataFrame(userinput, index=[0])
        transformed_userinput = preprocessor.transform(df_userinput)

        # load model and predict
        model = keras.models.load_model('../models/model.h5', compile=False)
        prediction_log_return = model.predict(transformed_userinput).flatten()[0]
        return {'prediction': float(prediction_log_return)}


if __name__ == '__main__':
    prediction = predict(user_year=2022, user_month=6, user_day=21, user_postcode='N1 2JU', user_property_type='F', user_property_age='N', user_ground='F')
    print(f'The prediction for the default values is: {prediction}')

# sample_data = {
#     'year': [2022],
#     'month': [6],
#     'day': [21],
#     'postcode': [N1 2JU],
#     'property_type': ['F'],
#     'property_age': ['N'],
#     'ground': ['F']
# }
