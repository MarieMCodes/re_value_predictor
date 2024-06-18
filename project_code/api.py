from fastapi import FastAPI
import pickle

app = FastAPI()

@app.get('/')
def root():
    return {'greeting': "hello"}


@app.get('/predict')
def predict(free_text_address):
    pass
    # with open ('../models/best_model.pkl', 'rb') as model_file:
    #     model = pickle.load(model_file)
    # codified_address = placeholder_function_to_codify_text(free_text_address)
    # prediction = model.predict(codified_text) #this means we expect
    # return prediction
