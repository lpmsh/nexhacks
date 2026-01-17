from typing import Any, Dict
from pydantic import ValidationError
from fastapi import HTTPException

from ..schemas.market import Market
from ..schemas.insight import Insight

class APIHandler:
    @staticmethod
    def root() -> Dict[str, Any]:
        return {"message": "Hello from Nexhacks API!"}

    @staticmethod
    def health() -> Dict[str, Any]:
        return {"status": "healthy"}

    @staticmethod
    def create_market(payload: Any) -> Dict[str, Any]:
        try:
            m = Market.parse_obj(payload)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        # business logic placeholder (persist, call services, etc.)
        return {"message": f"Market {m.market} created successfully"}

    @staticmethod
    def create_insight(payload: Any) -> Dict[str, Any]:
        try:
            ins = Insight.parse_obj(payload)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        # business logic placeholder (persist, call services, etc.)
        return {"message": f"Insight for {ins.market} accepted"}
