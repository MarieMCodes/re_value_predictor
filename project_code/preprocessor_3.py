import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer, make_column_selector

def feature_target(df: pd.DataFrame):
    df = df[(df['price'] >= 300000) & (df['price'] <= 1500000)]
    #The log return at time step t is equal to log(price(t)/price(t-1)).Then model could predict a change in price from the most recent time step.
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna(subset=['log_return'])
    # Define the target and features
    y = df['log_return']
    X = df.drop(columns=['log_return', 'price'])
    return X, y


def preprocess_fit_X(X: pd.DataFrame):
    """
    takes X as df and preprocess - fits all columns without transforming.
    Output is a fitted columntransformer (processor) that can transform any X df
    """
    X.columns = X.columns.str.strip()
    numeric_pipeline = Pipeline([('scaler', RobustScaler())])

    categorical_pipeline = Pipeline([
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
    ], remainder='passthrough') 

    preprocessor_fitted = preprocessor.fit(X)

    return preprocessor_fitted

