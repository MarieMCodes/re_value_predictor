import pandas as pd
import numpy as np

"""
This script load and sample the dataset
"""
    
# columns = ['price', 'date', 'postcode', 'property_type', 'property_age', 'ground',
#       'number', 'additional_info', 'street', 'locality', 'town', 'district',
#       'county', 'year', 'month', 'day'],
#

file='../raw_data/london_real_estate_data.zip'
df=pd.read_csv(file,
               compression='zip',
               dtype={'price': np.int32,'day':np.int16, 'month':np.int16,'year':np.int16}
)

def load_sample_data(df: pd.DataFrame, n_samples: int = 5000, random_state: int = 1) -> pd.DataFrame:
    """
    Load a sample of the dataset.
    """
    df_sampled = df.sample(n=n_samples, random_state=random_state)
    return df_sampled