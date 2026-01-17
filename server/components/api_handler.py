from typing import Any, Dict
from pydantic import ValidationError
from fastapi import HTTPException

from schemas.market import Market


class APIHandler:
    """Central handler that validates payloads and performs endpoint logic."""

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
        return {"message": f"Market {m.market} created successfully"}
