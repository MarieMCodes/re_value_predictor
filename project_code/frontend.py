import streamlit as st
import requests

# web page title
st.title('London Real Estate Value Predictor')

# get our free text address input
text = st.text_input(label='Enter your London address here', value='')

# Method 1 - local
url = 'http://127.0.0.1:8000/predict'

# Method 2 - on streamlit
# url = 'https://basic-api-32vj2qxrpq-ew.a.run.app/predict' #here goes the google run container instance API URL

params = {
    'free_text_address': str(text)
}
if text:
    response = requests.get(url=url, params=params).json()['prediction']
    st.write(response)
else:
    st.write('Awaiting input...')
