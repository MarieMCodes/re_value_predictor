import pandas as pd
import numpy as np
import requests

from project_code.data import load_csv, tidy_df

### WORK IN PROGRESS

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

    # create new df with postcode nans
    postcode_nans=df[df['postcode']==np.nan]

    # drop postcode nans
    df= df.dropna(subset=['postcode'])

    # make api call
    df['lat_lon']=df['postcode'].apply(postcode_api)

    # separate out rows with Errors and add to postcode_nans
    errors=df[df['lat_lon']=='ERROR']

    # create new lat lon columns
    df['lat']=df['lat_lon'].apply(lambda x: x[0])
    df['lon']=df['lat_lon'].apply(lambda x: x[1])

    # write to csv
    df.to_csv('london_re_postcodes_latlon.zip', compression='zip',Index=False, float_format='%.7f')

    # merge errors df w postcode nan df - append rows
    missing_values=errors.concat(postcode_nans, axis=0)

    # write errors to csv
    missing_values.to_csv('london_re_postcodes_latlon_errors.zip', compression='zip',Index=False, float_format='%.7f')

    return df, missing_values



#### Run openstreet api to get lat long for address


def get_coordinates(address):
    """
    makes api call to openstreet to get lat lon and address based on address
    """
    url="https://nominatim.openstreetmap.org/"
    try:
        params={"q": address, "format": "json"}
        response=requests.get(url,params=params).json()
        lat=response[0]["lat"]
        lon=response[0]["lon"]
        full_address=response[0]["display_name"]
        return lat, lon, full_address
    except:
        return "ERROR"


def address_api_call():
    """
    loads raw data, cleans, create new 'address' column for api call,
    makes openstreet address api call to retrieve lat lon address per address,
    creates new df with lat lon address columns and writes data to csv,
    and one df for the error columns and writes to csv
    """
    data=load_csv()
    df=tidy_df(data)
    # create address column for api call
    df['address']=df['number']+ " " + df['street'] + " " + df['borough'] + " London"

    # make api call
    df['lat_lon_a']=df['address'].apply(get_coordinates)

    # separate out rows with Errors
    errors=df[df['lat_lon_a']=='ERROR']

    # create new lat lon api_address columns
    df['lat']=df['lat_lon_a'].apply(lambda x: x[0])
    df['lon']=df['lat_lon_a'].apply(lambda x: x[1])
    df['api_address']=df['lat_lon_a'].apply(lambda x: x[2])

    # write to csv
    df.to_csv('london_re_address_latlon.zip', compression='zip',Index=False, float_format='%.7f')

    # write to csv
    errors.to_csv('london_re_address_latlon_errors.zip', compression='zip',Index=False, float_format='%.7f')

    return df, errors
