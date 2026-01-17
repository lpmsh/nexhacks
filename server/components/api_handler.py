import os
from typing import Any, Dict

from fastapi import HTTPException
from supabase import Client, create_client
from dotenv import load_dotenv

from schemas.market import Market


class APIHandler:
    def __init__(self) -> None:
        load_dotenv()
        url: str = os.getenv("SUPABASE_URL", "")
        key: str = os.getenv("SUPABASE_SECRET_KEY", "")
        
        if url == "" or key == "":
            raise HTTPException(
                status_code=500,
                detail="Supabase configuration missing (SUPABASE_URL / SUPABASE_SECRET_KEY).",
            )

        self.supabase: Client = create_client(url, key)
    
    def root(self) -> Dict[str, str]:
        return {"message": "Welcome to the Nexhacks API!"}
    
    def health(self) -> Dict[str, str]:
        return {"status": "healthy"}

    def create_market(self, market: Market) -> Dict[str, Any]:
        try:
            resp = (self.supabase.table("markets")
                    .insert({"market": market.market_name})
                    .execute()
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Supabase request failed: {str(exc)}"
            )

        return {"data": resp.data}