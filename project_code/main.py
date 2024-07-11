import pandas as pd
import numpy as np
from preprocessor import preprocess_fit_X, split_data, feature_target
from model import initialize_model, compile_model, train_model, evaluate_model
#from tensorflow.keras import model
from tensorflow.keras.models import save_model, load_model
import pickle



## INITIAL TRAINING
# load data - sample first
# master data columns: ['price', 'date', 'property_type', 'property_age',
# 'ownership', 'year', 'month', 'lat', 'lon', 'sin_time', 'cos_time']
def run_initial_training():
    """
    runs initial training: from loading data, processing it, splitting data,
    instantiating model, compiling, training and evaluating.
    Output MAE and MSE
    """
    # data is already stripped of properties <200k and above 15Ml
    data=pd.read_csv('../raw_data/london_re_postcodes_latlon_master.zip',
                    dtype={'price': np.int32,'month':np.int16,'year':np.int16},
                    )#.sample(400000)
    # master: drop month, drop date
    data.drop(columns=['date','month'],inplace=True)

    # remove premium price outliers
    clean_data=data[(data['price']<2500000) & (data['price']>199000)]
    print('✅ data loaded and cleaned')

    X,y=feature_target(clean_data)
    # split the X and y
    X_train, X_test, y_train, y_test= split_data(X,y)
    print('✅ data split')

    # fit preprocessor on training set
    preprocessor_fitted=preprocess_fit_X(X_train)
    print(" ✅ X_train fitted and saved preprocessor")

    # Export processor as pickle file
    with open("../models/preprocessor_final.pkl","wb") as file:
        pickle.dump(preprocessor_fitted, file)

    # transfrom X_train,X_test
    X_train_processed= preprocessor_fitted.transform(X_train)
    X_test_processed= preprocessor_fitted.transform(X_test)
    print(f'✅ X_train processed with shape {X_train_processed.shape}')
    print(f'✅ X_test processed with shape {X_test_processed.shape}')


    # initialise_model
    model=initialize_model()

    # compile_model
    model=compile_model(model)

    # train_model
    model=train_model(model,X_train_processed,y_train)[0]
    print('✅ model finished training')

    #save model
    save_model(model, '../models/model_final.h5')
    print('✅ model saved ')

    # evaluate
    mae, mse=evaluate_model(model,X_test_processed,y_test)
    print(f'MAE is {mae}, and MSE is {mse}. Training is complete')
    return  mae,mse


# prediction:
#Xnew features='property_type', 'property_age',
# 'ownership', 'year', 'lat', 'lon', 'sin_time', 'cos_time']

# will stil need to create a converter from address (or postcode)
# to lat lon and month to sin cos
def prediction(year=2026,
    property_type='F',
    property_age='N',
    ownership='L',
    lat=51.5487553,
    lon=-0.1235217,
    sin_time=-0.5,
    cos_time=-0.866025):
    """
    takes new X, processes them and predicts
    """
    X_new=pd.DataFrame(locals(),index=[0])
    print(f"✅ new data loaded in df with shape {X_new.shape}")

    # load preprocessor
    preprocessor=pickle.load(open("../models/preprocessor_final.pkl","rb"))
    X_new_processed=preprocessor.transform(X_new)
    print(f"✅ new data processed with shape {X_new_processed.shape}")

    #load model
    model=load_model('../models/model_final.h5')
    print("✅ model loaded")

    # predict
    ypred=model.predict(X_new_processed)
    print("✅ model predicted")

    # reverse log price to actual price
    prediction=np.exp(ypred)
    print(f" Your predicted price for the property in {year} is: {prediction}")
    return prediction

# y = new build  n = old building
if __name__ == '__main__':
    #run_initial_training()
    prediction = prediction()
    # year=2023,
    # property_type='F',
    # property_age='N',
    # ownership='L',
    # lat=51.491539,
    # lon=0.026218,
    # sin_time=0.5,
    # cos_time=0.85
    # )
