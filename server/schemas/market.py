from pydantic import BaseModel


class Market(BaseModel):
    market_name: str