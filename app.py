from fastapi import FastAPI, Request
import uvicorn

import json

from custom_rul_prediction import infer


app = FastAPI()

@app.post("/")
async def stream(request: Request):
    data = await request.json()
    data = json.dumps(data)
    pred = infer(data)
    return {"prediction": pred}


if __name__ == "__main__":
    uvicorn.run("app:app", port=8002, reload=True)