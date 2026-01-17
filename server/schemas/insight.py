from pydantic import BaseModel

class Insight(BaseModel):
    market: str
    insight_medium: str 
    insight_text: str