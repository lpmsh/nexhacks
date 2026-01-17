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
    def create_market(market: Market) -> Dict[str, Any]:
        
        return {"message": f"Market {market.market_name} created successfully"}
