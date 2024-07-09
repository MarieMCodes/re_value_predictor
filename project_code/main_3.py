# main.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model_3 import process_and_train, evaluate_model, predict_lat_lon, build_dense_model
from preprocessor_3 import feature_target, preprocess_fit_X
from tensorflow.keras.models import load_model

file_path = '../raw_data/london_re_postcodes_latlon_master.zip'
chunk_size = 100000


def run_pipeline(file_path, chunk_size, batch_size=64):
    # Load the data
    data = pd.read_csv(file_path, dtype={'price': np.int32, 'month': np.int16, 'year': np.int16})
    data.drop(columns=['date', 'month'], inplace=True)

    # Process each chunk and train the model
    model, preprocessor_fitted, scaler_y, X_test, y_test_scaled = process_and_train(file_path, chunk_size, batch_size=batch_size)

    # Make predictions on the test set
    y_pred_scaled = model.predict(X_test).flatten()

    # Rescale the predictions and actual test values
    y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test_scaled).flatten()

    # Save the model and preprocessor
    model.save('../models/model_3.h5')
    with open('../models/preprocessor_last.pkl', 'wb') as file:
        pickle.dump(preprocessor_fitted, file)

    print('✅ model saved ')
    
    # Evaluate the model
    mae, mse = evaluate_model(model, X_test, y_test_rescaled)
    print(f'MAE is {mae}, and MSE is {mse}. Training is complete')

def example_prediction(file_path, default_values, lat, lon):
    df = pd.read_csv(file_path, compression='zip')
    initial_price = df[(df['lat'] == lat) & (df['lon'] == lon)]['price'].iloc[-1]

    # Load the model and preprocessor
    loaded_model = load_model('../models/model_3.h5')
    with open('preprocessor_last.pkl', 'rb') as file:
        loaded_preprocessor = pickle.load(file)

    print("✅ model loaded")

    # Predict using the reloaded model and preprocessor
    predicted_value_loaded = predict_lat_lon(loaded_model, loaded_preprocessor, scaler_y, lat, lon, default_values, initial_price)
    print(f"Predicted value for lat {lat} and lon {lon} with loaded model: {predicted_value_loaded}")

    return predicted_value_loaded

if __name__ == '__main__':
    file_path = '../raw_data/london_re_postcodes_latlon_master.zip'
    chunk_size = 100000
    
    run_pipeline(file_path, chunk_size, batch_size=64)
    
    # Example default values for prediction
    default_values = {
        'sin_time': 0.5,
        'ownership': 'F',
        'property_age': 'N',
        'cos_time': -0.5,
        'year': 2024,
        'property_type': 'T'
    }
    
    lat = 51.546844
    lon = -0.093986
    
    example_prediction(file_path, default_values, lat, lon)
