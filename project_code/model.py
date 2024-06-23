import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))  # Output layer
    return model

def compile_model(X_train, y_train, X_val, y_val, input_shape: tuple, epochs=50, batch_size=32) -> (Model, dict):
    model = initialize_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model, history

def evaluate_model(model: Model, X_test: np.ndarray, y_test: np.ndarray) -> (float, float):
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    return mae, rmse

def plot_predictions(y_test: np.ndarray, y_pred: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()

def cross_validate_model(X: np.ndarray, y: np.ndarray, input_shape: tuple, epochs=50, batch_size=32, n_splits=5) -> (list, list):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = initialize_model(input_shape)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred = model.predict(X_val).flatten()
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

    print(f"Cross-Validation MAE: {np.mean(mae_scores)} (+/- {np.std(mae_scores)})")
    print(f"Cross-Validation RMSE: {np.mean(rmse_scores)} (+/- {np.std(rmse_scores)})")
    return mae_scores, rmse_scores
