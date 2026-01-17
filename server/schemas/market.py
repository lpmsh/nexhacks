from pydantic import BaseModel


class Market(BaseModel):
    market: str