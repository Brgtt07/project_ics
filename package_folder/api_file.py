from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {'hello': 'world'}

@app.get('/predict')
def predict():

    return {'hello': 'world'}
