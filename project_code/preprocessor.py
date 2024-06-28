import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def feature_target(df: pd.DataFrame):
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna(subset=['log_return'])

    # Define the target and features
    y = df['log_return']
    X = df.drop(columns=['log_return', 'price'])
    return X, y 

def preprocess_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, ColumnTransformer, RobustScaler):
    X, y = feature_target(df)  # Get the features and target
    
    X.columns = X.columns.str.strip()
    
    # Define numerical and categorical columns
    numeric_columns = ['year', 'month', 'day']
    categorical_columns = ['postcode', 'property_type', 'property_age', 'ground']

    # Preprocessing pipelines for numerical and categorical data
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column transformer to preprocess both numerical and categorical data
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])   
    
    X_transformed = preprocessor.fit_transform(X)
    
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    
    scaler_y = RobustScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return X_transformed, y_scaled, preprocessor, scaler_y

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.2):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
