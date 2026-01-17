from fastapi import FastAPI, Body
from components.api_handler import APIHandler

app = FastAPI(
    title="Nexhacks API",
    description="Backend API for Nexhacks",
    version="0.1.0",
)


@app.get("/")
async def root():
    return APIHandler.root()


@app.get("/health")
async def health_check():
    return APIHandler.health()


@app.post("/new/market")
async def create_market(payload: dict = Body(...)):
    return APIHandler.create_market(payload)
