from fastapi import FastAPI
import pickle

import pandas as pd
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import load_model


app = FastAPI()

# load model into cache (memory) of uvicorn
app.state.model = load_model('../models/london_re_model_latlon_full.h5')

@app.get('/')
def root():
    return {'greeting': "hello"}


@app.get('/predict')
def predict(year: int,
            property_type: str,
            property_age: str,
            ownership: str,
            lat: float,
            lon: float,
            sin_time: float,
            cos_time: float):

    # load user input in correct format
    X_user = pd.DataFrame(locals(),index=[0])

    #load preprocessor
    preprocessor=pickle.load(open("../models/preprocessor_latlon.pkl","rb"))

    #preprocess user input
    X_user_processed=preprocessor.transform(X_user)

    # load cached model and predict
    model=app.state.model
    y_pred=model.predict(X_user_processed)
    prediction=np.exp(y_pred)[0][0]

    return {'prediction in £': int(prediction)}


if __name__ == '__main__':
    prediction = predict(year=2023,
                         property_type='F',
                         property_age='O',
                         ownership='L',
                         lat=51.491539,
                         lon=0.026218,
                         sin_time=0.5,
                         cos_time=0.85)
    print(f'The prediction for the default values is: {prediction}')
