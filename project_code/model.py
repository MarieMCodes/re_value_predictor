import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error


def initialize_model() -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    return model


def compile_model(model:Model, learning_rate= 0.001, epochs= 100, batch_size=32) -> tuple[Model, dict]:
    """
    compiles model
    """
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])
    print("✅ Model compiled")
    return model


def train_model(
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size=32,
        patience=10,
        validation_data=None,
        validation_split=0.3
    ) -> tuple[Model, dict]:
    """
    trains model
    """
    print("starting model training")

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X_train)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history




def evaluate_model(model: Model, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """
    evaluate trained model on test data
    """
    metrics = model.evaluate(X_test,
                             y_test,
                             batch_size=32,
                             return_dict=True
    )

    mse = metrics["loss"]
    mae = metrics["mae"]
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    return mae, mse
