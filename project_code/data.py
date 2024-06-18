import pandas as pd
import numpy as np

# columns of loaded df= ['price', 'date', 'postcode', 'property_type', 'property_age', 'ground',
#       'number', 'additional_info', 'street', 'locality', 'town', 'district',
#       'county', 'year', 'month', 'day'],
#

def load_csv():
    ''' loads london csv file from raw_data folder '''
    file='../raw_data/london_real_estate_data.zip'
    df=pd.read_csv(file,
                compression='zip',
                dtype={'price': np.int32,'day':np.int16, 'month':np.int16,'year':np.int16}
    )
    return df



# columns of tidy df= ['price', 'date', 'postcode', 'property_type', 'property_age', 'ground',
#       'street', 'borough', 'year', 'month', 'day', 'full_property_number']
#

def tidy_df(df):
    ''' takes london re data df and deletes locality, town, county columns,
    merges number and additional info to create new column and deleting the former individual ones,
    and renames district column to borough.'''
    # replace nan values w empty strings
    df.fillna({'additional_info':''}, inplace=True)

    # merging number and additional number
    df['full_property_number']=df['number']+ '' + df['additional_info']

    # dropping columns
    df.drop(columns=['number', 'additional_info', 'locality','town','county'], inplace=True)

    df.rename(columns={'district':'borough'}, inplace=True)

    return df
