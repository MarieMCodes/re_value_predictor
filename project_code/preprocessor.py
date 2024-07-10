import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
#from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

def feature_target(df: pd.DataFrame):
    #df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df['log_return'] = np.log(df['price'])
    df = df.dropna(subset=['log_return'])

    # Define the target and features
    y = df['log_return']
    X = df.drop(columns=['log_return', 'price'])
    return X, y


def preprocess_fit_X(X: pd.DataFrame):
    """
    takes X as as df and preprocess - fits all columns without transforming.
    Output is a fitted columntransformer (processor) that can transform any X df
    """
    X.columns = X.columns.str.strip()

    # Define numerical and categorical columns
    #numeric_columns = ['year','sin_time','cos_time'] # ,'lat','lon']   keeping those as is and not scaling them
    #categorical_columns = ['property_type', 'property_age', 'ownership']

    # Preprocessing pipelines for numerical and categorical data
    numeric_pipeline = Pipeline([('scaler', RobustScaler())])

    categorical_pipeline = Pipeline([
        #('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',
                                 drop='if_binary',
                                 sparse_output=False
                                 ).set_output(transform='pandas')
        )
    ])

    # Column transformer to preprocess both numerical and categorical data
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, make_column_selector(dtype_exclude="object")),
        ('cat', categorical_pipeline, make_column_selector(dtype_include="object"))
    ], remainder='passthrough') # no need to scale lat lon

    preprocessor_fitted = preprocessor.fit(X)

    return preprocessor_fitted


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """
    splits X and y into X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test




# def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.2):
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

#     return X_train, X_val, X_test, y_train, y_val, y_test
