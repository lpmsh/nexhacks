from fastapi import FastAPI
from .schemas.market import Market

app = FastAPI(
    title="Nexhacks API",
    description="Backend API for Nexhacks",
    version="0.1.0",
)

@app.get("/")
async def root():
    return {"message": "Hello from Nexhacks API!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.put("/new/market")
async def create_market(market: Market):
    return {"message": f"Market {market.market} created successfully"}