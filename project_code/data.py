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
    file='../raw_data/london_re_address_latlon_sample.zip'
    df=pd.read_csv(file,
                compression='zip',
                dtype={'price': np.int32,'day':np.int16, 'month':np.int16,'year':np.int16}
    )
    return df




# columns of output df=['price', 'year', 'month', 'lat', 'lon', 'property_type','property_age', 'ownership']
def load_tidy_latlong_sample():
    """
    loads csv w clean sample raw data (7T rows) w lat lon columns.
    - total of 8 columns:
    price, date, lat, lon, property_type, property_age, ownership (formerly ground),address
    - dropping date and address columns
    - has index
    - output df has 6 columns
    """
    file='../raw_data/london_real_estate_data.zip'
    df=pd.read_csv(file,
               compression='zip',
               dtype={'price': np.int32, 'month':np.int16,'year':np.int16, 'lat':np.float64, 'lon':np.float64}
                  )

    df.drop(labels=['date','address'], axis='columns', inplace=True)
    return df




# columns of tidy_df= ['price', 'date', 'postcode', 'property_type', 'property_age', 'ownership',
#     'year', 'month']

def tidy_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    takes raw df and:
    tidies some nan columns,
    renames district and ground columns,
    '''
    # replace nan values w empty strings
    df.fillna({'additional_info':''}, inplace=True)

    # merging number and additional number
    df['full_property_number']=df['number']+ ' ' + df['additional_info']

    df.rename(columns={'district':'borough', 'ground':'ownership'}, inplace=True)

    # dropping rows where 'type'=='O'
    df=df[df['property_type']!='O']

    # dropping rows where 'ownership' == 'U'
    df=df[df['ownership']!='U']

    return df


def shorten_df(df):
    """
    removes unneccessary columns and postcode nan values,
    remove price outliers
    """
    # dropping columns
    df.drop(columns=['day', 'number', 'street', 'additional_info',\
        'full_property_number', 'borough', 'locality','town','county'], inplace=True)

    # drop rows with NaN (3000 in postcode)
    df.dropna(axis=0, how='any', inplace=True)

    #remove price outliers
    df=df[df['price']>199_999]
    df=df[df['price']<15_000_000]

    return df


# NOT NEEDED FOR NOW as we are working with the full postcode!!
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''
    taking dataframe and adding new columns:
    - 2 and 3 chars shortened postcodes,
    - sin cos for months
    returns df
    '''
    # add shortened postcodes - choose one of the the below
    df['short2_pc']=df[['postcode']].apply(lambda x: x.str[:2])
    #df['short3_pc']=df[['postcode']].apply(lambda x: x.str[:3])

    # dropping postcode
    # df.drop(columns='postcode', inplace=True)

    # add sin cos feature engineering

    return df
