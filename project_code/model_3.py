import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

from preprocessor_3 import preprocess_fit_X, feature_target

def build_dense_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))  # Output layer
    return model


def process_and_train(file_path, chunk_size, model=None, preprocessor=None, batch_size=64):
    scaler_y = StandardScaler()
    y_scaler_fitted = False
    
    for i, chunk in enumerate(pd.read_csv(file_path, compression='zip', chunksize=chunk_size)):
        X, y = feature_target(chunk)
        
        if preprocessor is None:
            preprocessor = preprocess_fit_X(X)
        
        X_transformed = preprocessor.transform(X)
        
        # for DNN model
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Scale target values
        if not y_scaler_fitted:
            y_train_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
            y_scaler_fitted = True
        else:
            y_train_scaled = scaler_y.transform(y.values.reshape(-1, 1))

        # Determine the input shape from the preprocessed data
        if model is None:
            input_shape = (X_transformed.shape[1],)
            model = build_dense_model(input_shape)
            model.compile(optimizer=RMSprop(learning_rate=0.01), loss='mean_squared_error', metrics=['mae'])
        
        # Split the data
        X_train, X_val_test, y_train, y_val_test = train_test_split(X_transformed, y_train_scaled, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.2, random_state=42)

        # Train the model on this chunk
        es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1)
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[es], verbose=1)
    
    return model, preprocessor, scaler_y, X_test, y_test

def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=0)
    return results[1], results[0]

def predict_lat_lon(model, preprocessor, scaler_y, lat, lon, default_values, initial_price):
    new_data = pd.DataFrame({
        'lat': [lat],
        'lon': [lon],
        'sin_time': [default_values['sin_time']],
        'ownership': [default_values['ownership']],
        'property_age': [default_values['property_age']],
        'cos_time': [default_values['cos_time']],
        'year': [default_values['year']],
        'property_type': [default_values['property_type']]
    })
    new_data_transformed = preprocessor.transform(new_data)
    prediction_log_return = model.predict(new_data_transformed).flatten()[0]
    y_pred_rescaled = scaler_y.inverse_transform([[prediction_log_return]])
    predicted_price = initial_price * np.exp(y_pred_rescaled.flatten()[0])
    return predicted_price

