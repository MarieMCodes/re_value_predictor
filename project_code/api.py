from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {'greeting': "hello"}


@app.get('/predict')
def predict():
    pass
