from fastapi import FastAPI, Query

app = FastAPI()

@app.get('/')
def root():
    return {'hello': 'world'}

@app.get('/predict')
def predict(
    greeting: str = Query(..., description="A greeting message")
):
    if greeting == "hello":
        return {"response": "banana"}
    else:
        return {"response": f"You said: {greeting}"}


@app.get('/top_countries_weighted_sum')
def top_countries_weighted_sum(
    #define the expected inputs
):
    return {"response": "banana"}