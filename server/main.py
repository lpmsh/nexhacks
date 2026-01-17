from fastapi import FastAPI

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
