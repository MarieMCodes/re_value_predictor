import pandas as pd
import numpy as np
from preprocessor import preprocess_data, split_data, preprocess_features,feature_target
from model_2 import initialize_model, compile_model, train_model, evaluate_model
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
    data=pd.read_csv('../raw_data/london_re_postcodes_latlon_master.zip',
                    dtype={'price': np.int32,'month':np.int16,'year':np.int16},
                    ).sample(100000)
    # master: drop month, drop date
    data.drop(columns=['date','month'],inplace=True)
    print('✅ data loaded')

    X,y=feature_target(data)
    # preprocess whole df
    preprocessor_fitted=preprocess_data(X)

    X_processed= preprocessor_fitted.transform(X)
    # Export processor as pickle file
    with open("../models/preprocessor_latlon.pkl", "wb") as file:
	    pickle.dump(preprocessor_fitted, file)


    print(f'✅ data processed with shape (X) {X_processed.shape}')

    # split the processed X and y
    X_train, X_test, y_train, y_test= split_data(X_processed,y)
    print('✅ data split, next step initialising model')

    # initialise_model
    model=initialize_model()

    # compile_model
    model=compile_model(model,learning_rate=0.001, epochs=80 )

    # train_model
    model=train_model(model,X_train,y_train)[0]
    print('✅ model finished training')

    #save model
    model.save('../models/london_re_model_latlon_sample')
    print('✅ model saved ')

    # evaluate
    mae, mse=evaluate_model(model,X_test,y_test)
    print(f'MAE is {mae}, and MSE is {mse}. Training is complete')
    return  mae,mse


# prediction:
#Xnew features='property_type', 'property_age',
# 'ownership', 'year', 'lat', 'lon', 'sin_time', 'cos_time']

# will stil need to create a converter from address (or postcode)
# to lat lon and month to sin cos
def prediction(year=2023,
    property_type='F',
    property_age='O',
    ownership='L',
    lat=51.491539,
    lon=0.026218,
    sin_time=0.5,
    cos_time=0.85):
    """
    takes new X, processes them and predicts
    """
    X_new=pd.DataFrame(locals(),index=[0])
    print(f"✅ new data loaded in df with shape {X_new.shape}")

    # load preprocessor
    preprocessor=pickle.load(open("../models/preprocessor_latlon.pkl","rb"))
    X_new_processed=preprocessor.transform(X_new)
    print(f"✅ new data processed with shape {X_new_processed.shape}")

    #load model
    model=load_model('../models/london_re_model_latlon_sample')
    print("✅ model loaded")

    # predict
    ypred=model.predict(X_new_processed)
    print("✅ model predicted")

    # reverse log price to actual price
    prediction=np.exp(ypred)
    print(f" Your predicted price for the property is: {prediction}")
    return prediction


if __name__ == '__main__':
    run_initial_training()
    prediction = prediction(year=2023,property_type='F', property_age='O',
    ownership='L',
    lat=51.491539,
    lon=0.026218,
    sin_time=0.5,
    cos_time=0.85
    )
