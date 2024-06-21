import pandas as pd
import numpy as np

# columns = ['price', 'date', 'postcode', 'property_type', 'property_age', 'ground',
#       'number', 'additional_info', 'street', 'locality', 'town', 'district',
#       'county', 'year', 'month', 'day']

def load_sample_data(df: pd.DataFrame, n_samples: int = 5000, random_state: int = 1) -> pd.DataFrame:
    """
    Load a sample of the dataset.
    """
    file='../raw_data/london_real_estate_data.zip'
    df_sampled=pd.read_csv(file,
               compression='zip',
               dtype={'price': np.int32,'day':np.int16, 'month':np.int16,'year':np.int16}
                  ).sample(n=n_samples, random_state=random_state)
    return df_sampled



def load_csv():
    ''' loads london csv file from raw_data folder '''
    file='../raw_data/london_real_estate_data.zip'
    df=pd.read_csv(file,
                compression='zip',
                dtype={'price': np.int32,'day':np.int16, 'month':np.int16,'year':np.int16}
    )
    return df



# columns of tidy_df= ['price', 'date', 'postcode', 'property_type', 'property_age', 'ground',
#       'street', 'borough', 'year', 'month', 'day', 'full_property_number']

def tidy_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    takes london re df and deletes locality, town, county columns,
    merges number and additional info to create new column and deleting the former individual ones,
    and renames district column to borough.
    '''

    # replace nan values w empty strings
    df.fillna({'additional_info':''}, inplace=True)

    # merging number and additional number
    df['full_property_number']=df['number']+ '' + df['additional_info']

    # dropping columns
    df.drop(columns=['number', 'additional_info', 'locality','town','county'], inplace=True)

    df.rename(columns={'district':'borough'}, inplace=True)

      # dropping rows where 'type'=='O'
    df=df[df['property_type']!='O']

    # dropping rows where 'ground' == 'U'
    df=df[df['ground']!='U']

    return df



def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''
    taking dataframe and adding new columns:
    - 2 and 3 chars shortened postcodes,
    - sin cos for months
    returns df
    '''

    # add shortened postcodes - choose of the the below
    df['short2_pc']=df[['postcode']].apply(lambda x: x.str[:2])
    #df['short3_pc']=df[['postcode']].apply(lambda x: x.str[:3])

    # dropping postcode
    df.drop(columns='postcode', inplace=True)

    # add sin cos feature engineering

    return df
