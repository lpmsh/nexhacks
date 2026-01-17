from fastapi import FastAPI
from components.api_handler import APIHandler
from schemas.market import Market

app = FastAPI(
    title="Nexhacks API",
    description="Backend API for Nexhacks",
    version="0.1.0",
)

api_handler = APIHandler()

@app.get("/")
async def root():
    return api_handler.root()

@app.get("/health")
async def health_check():
    return api_handler.health()


@app.post("/new/market")
async def create_market(market: Market):
    return api_handler.create_market(market)