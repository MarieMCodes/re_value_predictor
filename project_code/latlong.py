import pandas as pd
import numpy as np
import requests

#from project_code.data import load_csv, tidy_df

### For Virtual Machine
def load_csv():
    ''' loads london csv file from raw_data folder '''
    file='raw_data/london_real_estate_data.zip'
    df=pd.read_csv(file,
                compression='zip',
                dtype={'price': np.int32,'day':np.int16, 'month':np.int16,'year':np.int16}
    )
    return df


def tidy_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    takes london re df and deletes all columns not needed as features for model,
    renames district and borough columns,
    removes nan values,
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



#### Run Postcodes.io api to get lat long for postcodes

def postcode_api(postcode):
    """
    makes api call to postcodes.io to get lat lon based on postcode
    """
    url=f"https://api.postcodes.io/postcodes/{postcode}"
    try:
        response=requests.get(url).json()
        #print (response)
        #print (response["result"])
        lon=response["result"]["longitude"]
        lat=response["result"]["latitude"]
        return lat, lon
    except:
        return "ERROR"



def postcodes_call():
    """
    loads raw data, cleans, separates out postcode nans,
    makes postcodes api call to retrieve lat lon per postcode,
    returns new df with lat lon columns and writes data to csv
    and returns one df for the error columns and writes to csv
    """
    data=load_csv()
    df=tidy_df(data)
    print(f"✅ data loaded and cleaned, df has {len(df)} rows")

    # create new df with postcode nans
    postcode_nans=df[df['postcode'].isna()==True]

    # drop postcode nans
    df=df[df['postcode'].isna()==False]
    print(f"👉{len(postcode_nans)} nans in postcode removed")
    print(f"✅  cleaned df has now {len(df)} rows")


    #Create batches to run api calls, starting most recent data first
    # first_chunk=df.iloc[3300000:,:] - DONE data retrieved
    second_chunk=df.iloc[3000000:3300000,:]
    third_chunk=df.iloc[2400000:3000000,:]
    fourth_chunk=df.iloc[1900000:2400000,:]
    fifth_chunk=df.iloc[1200000:1900000,:]
    sixth_chunk=df.iloc[500000:1200000,:]
    seventh_chunk=df.iloc[:500000,:]

    batches=[second_chunk,third_chunk,fourth_chunk,fifth_chunk,sixth_chunk, seventh_chunk]
    df_list=[]
    errors_list=[]

    # for loop to make api call
    for index,batch in enumerate(batches):
        # make api call
        print(f"👉 starting api calls for batch nr {index+2}")
        batch['lat_lon']=batch['postcode'].apply(postcode_api)
        print(f"✅ api calls for batch nr {index+2} done")

        # separate out rows with Errors and add to postcode_nans
        errors=batch[batch['lat_lon']=='ERROR']
        batch=batch[batch['lat_lon']!='ERROR']
        print(f"❗️ batch nr {index+2} contained {len(errors)} errors")
        print(f"👉 df has {len(batch)} clean rows ")

        # create new lat lon columns
        batch['lat']=batch['lat_lon'].apply(lambda x: x[0])
        batch['lon']=batch['lat_lon'].apply(lambda x: x[1])

        # write to csv
        batch.to_csv(f'london_re_postcodes_latlon_batch_{index+2}.zip', compression='zip',index=False, float_format='%.7f')
        print("✅ saved csv")

        # write errors to csv
        errors.to_csv(f'london_re_postcodes_latlon_batch_{index+2}_errors.zip', compression='zip',index=False, float_format='%.7f')
        print("✅ saved csv w errors")

        df_list.append(batch)
        errors_list.append(errors)

        print(f"✅ 🙌 batch nr {index+2} loop is done!")

    # merge errors df w postcode nan df - append rows
    # stack dfs together row wise. if error, try: ignore_index=True
    errors_list.append(postcode_nans)
    df_master=pd.concat(df_list,axis=0)
    errors_master=pd.concat(errors_list,axis=0)

    # write to csv
    df_master.to_csv('london_re_postcodes_latlon_all.zip', compression='zip',index=False)
    errors_master.to_csv('london_re_postcodes_latlon_all_errors.zip', compression='zip',index=False)

    print(f"✅ 🙌 df master merge complete and saved as csv!")

    return df_master, errors_master



#### Run openstreet api to get lat long for address

# removed full address for now to get faster api return...
def get_coordinates(address):
    """
    makes api call to openstreet to get lat lon based on address
    """
    url="https://nominatim.openstreetmap.org/search?"
    try:
        params={"q": address, "format": "json"}
        response=requests.get(url,params=params).json()
        lat=response[0]["lat"]
        lon=response[0]["lon"]
        #full_address=response[0]["display_name"]
        return lat, lon
    except:
        return "ERROR"



# https://nominatim.openstreetmap.org/q?Westminster%20Abbey&format=json

def address_api_call():
    """
    loads raw data, cleans, create new 'address' column for api call,
    makes openstreet address api call in batches to retrieve lat lon per address,
    creates new df with 'lat_lon_a' columns and writes data to csv,
    separate out lat lon
    and creates df for the error rows and writes to csv
    """
    data=load_csv()
    df=tidy_df(data)
    print(f"✅ data loaded and cleaned, df has {len(df)} rows")

    # create address column for api call
    df['address']=df['number']+ " " + df['street'] + " " + df['borough'] + " London"

    # remove NaNs and saves them to separate df
    nans=df[df['address'].isna()==True]
    print(f"👉{len(nans)} nans in address removed")

    nans.to_csv(f'london_re_address_latlon_nans.zip', compression='zip',index=False)
    print(f"👉 nans saved to csv")

    df=df[df['address'].isna()==False]
    print(f"✅ cleaned df has now {len(df)} rows")


    #Create batches to run api calls, starting most recent data first
    first_chunk=df.iloc[3300000:,:]
    second_chunk=df.iloc[2900000:3300000,:]
    third_chunk=df.iloc[2400000:2900000,:]
    fourth_chunk=df.iloc[1900000:2400000,:]
    fifth_chunk=df.iloc[1200000:1900000,:]
    sixth_chunk=df.iloc[500000:1200000,:]
    seventh_chunk=df.iloc[:500000,:]

    batches=[first_chunk,second_chunk,third_chunk,fourth_chunk,fifth_chunk,sixth_chunk, seventh_chunk]
    df_list=[]
    errors_list=[]
    # for loop to run through batches
    for index,batch in enumerate(batches):
        # make api call
        print(f"👉 starting api calls for batch nr {index+1}")
        batch['lat_lon_a']=batch['address'].apply(get_coordinates)

        # separate out two col and clean nans
        print(f"✅ api calls for batch nr {index+1} complete")
        batch['lat']=batch['lat_lon_a'].apply(lambda x: x[0])
        batch['lon']=batch['lat_lon_a'].apply(lambda x: x[1])
        errors=batch[batch['lat_lon_a']=='ERROR']
        print(f"❗️ batch nr {index+1} contained {len(errors)} errors")

        #drop errors and nans
        batch.dropna(inplace=True)
        batch=batch[batch['lat_lon_a']!='ERROR']

        # write to csv
        batch.to_csv(f'london_re_address_latlon_batch{index+1}.zip', compression='zip',index=False)
        print(f"✅ csv of batch nr {index+1} saved")

         # write errors to csv
        errors.to_csv(f'london_re_address_latlon_batch{index+1}_errors.zip', compression='zip',index=False, float_format='%.7f')
        print("✅ errors saved as csv")

        df_list.append(batch)
        errors_list.append(errors)
        print(f"✅ 🙌 batch nr {index+1} loop is done!")

    # stack dfs together row wise. if error, try: ignore_index=True
    errors_list.append(nans)
    df_master=pd.concat(df_list,axis=0)
    errors_master=pd.concat(errors_list,axis=0)

    # write to csv
    df_master.to_csv('london_re_address_latlon_all.zip', compression='zip',index=False)
    errors_master.to_csv('london_re_address_latlon_all_errors.zip', compression='zip',index=False)

    print(f"✅ 🙌 df master merge complete and saved as csv!")

    return df_master, errors_master
# afterwards still need to tidy saved dataframe, and remove unneccessary columns


# below code not working...?
if __name__ == '__main__':
    #postcodes_call()
    address_api_call
    print ('running address api call')
    #address='24 weymouth street, marylebone, london'
    #coordinates=get_coordinates(address)
    #print (f'the coordinates for {address} are: {coordinates}')
