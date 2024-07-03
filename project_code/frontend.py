import streamlit as st
import requests


def main(env='local'):
    # web page title
    st.title('London Real Estate Value Predictor')

    # we need inputs for year, month, postcode, property_type, property_age,ground

    # input year
    year = st.text_input(label='Select the year:', value='', autocomplete='202')
    #input sin_time
    sin_time = st.text_input(label='Sin Time:', value='')
    #input cos_time
    cos_time = st.text_input(label='Cos Time:', value='')
    #input property_type
    property_type = st.radio(label='Choose property type: F, T, S, D or O ', options=['F', 'T', 'S', 'D' ,'O'])
    #input property_age
    property_age = st.radio(label='Choose property age: O-Old, N-New', options=['O','N'])
    #input lat
    lat = st.number_input(label='Latitude')
    #input lon
    lon = st.number_input(label='Longitude')
    #input ownership
    ownership = st.text_input(label='Ownership')


    if year and sin_time and cos_time and ownership and property_type and property_age and lon and lat:
        params = {
            'year': year,
            'property_type': property_type,
            'property_age': property_age,
            'ownership': ownership,
            'lat': lat,
            'lon': lon,
            'sin_time': sin_time,
            'cos_time': cos_time
        }



        input_complete = True
        # Method 1 - local
        local_url_address_base = 'http://127.0.0.1:8000'

        # Method 2 - on streamlit
        online_url_address_base = 'https://basic-api-32vj2qxrpq-ew.a.run.app/predict' #here goes the google run container instance API URL

        # check if we run locally or on the cloud
        if env == 'local':
            url = f'{local_url_address_base}/predict'
        else:
            url = f'{online_url_address_base}/predict'


    try:
        if st.button("Model Prediction") and input_complete:
            # st.write(f'{url}/{params}')
            response = requests.get(url=url, params=params).json()
            nice_price = str(format(response['prediction in £'], ","))
            st.write('Sale Price Prediction: £', nice_price)
    except UnboundLocalError:
        st.write('No input or input incomplete.')

if __name__ == '__main__':
    main(env='local')
else:
    main(env='nonlocal')

# to be done:
# 1. find out how we can know if we run locally, then use that to alter the base_url part of the url above
# 2. make the frontend+api files work locally
# 3. build the local docker and check that it works locally with api + streamlit run from cli
# 4. build the online docker and check that it works online with api + streamlit run from cli
# 5. build the online docker and check that it works with online streamlit app
