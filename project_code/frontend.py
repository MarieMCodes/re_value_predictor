import streamlit as st
import requests

# web page title
st.title('London Real Estate Value Predictor')

# get our free text address input
free_text_address = st.text_input(label='Enter your London address here', value='')

# Method 1 - local
url = 'http://127.0.0.1:8000/'

# Method 2 - on streamlit
# url = 'https://basic-api-32vj2qxrpq-ew.a.run.app' #here goes the google run container instance API URL


response = requests.get(url=url).json()



st.write('The answer is', str(response))
