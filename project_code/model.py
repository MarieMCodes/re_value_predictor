import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(100, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="linear"))
    return model

def compile_model(model):
    """
    Complie the model
    """
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model


def train_model(model, train_generator, val_generator, epochs=100, batch_size=32):
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on the test dataset.
    """
    test_loss, test_mae = model.evaluate(test_generator)
    return test_loss, test_mae


def train_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train a baseline linear regression model and evaluate its performance.
    """
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    
    baseline_predictions_val = baseline_model.predict(X_val)
    baseline_mae_val = mean_absolute_error(y_val, baseline_predictions_val)
    print(f'Baseline Model MAE on validation set: {baseline_mae_val}')

    baseline_predictions_test = baseline_model.predict(X_test)
    baseline_mae_test = mean_absolute_error(y_test, baseline_predictions_test)
    print(f'Baseline Model MAE on test set: {baseline_mae_test}')