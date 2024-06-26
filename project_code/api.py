from fastapi import FastAPI
import pickle

import os
import pandas as pd
from keras.models import Sequential, Model


from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

app = FastAPI()

@app.get('/')
def root():
    return {'greeting': "hello"}


@app.get('/predict')
def predict(user_year, user_month, user_postcode, user_property_type, user_property_age, user_ground):
    userinput = {
        'user_year': user_year,
        'user_month': user_month,
        'user_postcode': user_postcode,
        'user_property_type': user_property_type,
        'user_property_age': user_property_age,
        'user_groun': user_ground
    }
    # print(userinput)
    # print(os.path.exists('../models/best_model.pkl'))
    with open ('../models/best_model.pkl', 'rb') as model_file:
        preprocessor, model = pickle.load(model_file)
        df_userinput = pd.DataFrame(userinput)
        transformed_userinput = preprocessor.transform(df_userinput)
        prediction_log_return = model.predict(transformed_userinput).flatten()[0]
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
    predict(user_year=1998, user_month=5, user_postcode='N1 2JU', user_property_type='F', user_property_age='O', user_ground='L')
