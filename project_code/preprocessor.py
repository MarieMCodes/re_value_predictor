import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def feature_target(df: pd.DataFrame):
    y=df["price"]
    X = df.drop(columns=["price"])
    return X, y 

def preprocess_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, ColumnTransformer, StandardScaler):
    X.columns = X.columns.str.strip()
    
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_columns),
            ('cat', categorical_pipeline, categorical_columns)
        ])
    
    X_transformed = preprocessor.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return X_transformed, y_scaled, preprocessor, scaler_y
    
def split_data(df: pd.DataFrame, split_ratio: float = 0.02) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    
    test_length = int(len(df) * split_ratio)
    val_length = int((len(df) - test_length) * split_ratio)
    train_length = len(df) - val_length - test_length

    df_train = df.iloc[:train_length, :].sample(frac=1, random_state=42)
    df_val = df.iloc[train_length:train_length + val_length, :].sample(frac=1, random_state=42)
    df_test = df.iloc[train_length + val_length:, :].sample(frac=1, random_state=42)

    return df_train, df_val, df_test


def prepare_generators(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, sequence_length: int = 10, batch_size: int = 32 ) -> (TimeseriesGenerator, TimeseriesGenerator, TimeseriesGenerator):

    train_generator = TimeseriesGenerator(X_train, y_train, length=sequence_length, batch_size=batch_size)
    val_generator = TimeseriesGenerator(X_val, y_val, length=sequence_length, batch_size=batch_size)
    test_generator = TimeseriesGenerator(X_test, y_test, length=sequence_length, batch_size=batch_size)

    return train_generator, val_generator, test_generator
