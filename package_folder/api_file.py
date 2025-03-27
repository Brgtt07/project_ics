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
