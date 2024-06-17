import streamlit as st
import requests

st.title('London Real Estate Value Predictor')


# Method 1
url = 'http://127.0.0.1:8000/'
# params = {'sepal_length': sep_len,
#           'sepal_width': sep_wid,
#           'petal_length': pet_len,
#           'petal_width': pet_wid}


response = requests.get(url=url).json()



st.write('The answer is', str(response))
