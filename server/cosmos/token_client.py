import os
from typing import Dict, Optional

from .scoring import build_metrics


class TokenCoClient:
    """Thin wrapper around tokenc SDK; optional to keep demos offline-friendly."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("TOKENC_API_KEY") or os.getenv("TTC_API_KEY")
        self._client = None
        self.available = False

        try:
            import tokenc  # type: ignore

            if self.api_key:
                self._client = tokenc.TokenClient(api_key=self.api_key)
                self.available = True
        except Exception:
            self.available = False

    def compress(
        self,
        text: str,
        aggressiveness: float = 0.5,
        max_output_tokens: Optional[int] = None,
        min_output_tokens: Optional[int] = None,
        model: str = "bear-1",
        api_key_override: Optional[str] = None,
    ) -> Dict:
        try:
            import tokenc  # type: ignore

            client = self._client
            if api_key_override:
                client = tokenc.TokenClient(api_key=api_key_override)
            if not client:
                return {"available": False, "error": "tokenc SDK unavailable or API key missing"}

            params = {
                "input": text,
                "aggressiveness": aggressiveness,
                "model": model,
            }
            if max_output_tokens is not None:
                params["max_output_tokens"] = max_output_tokens
            if min_output_tokens is not None:
                params["min_output_tokens"] = min_output_tokens
            response = client.compress_input(**params)
            compressed = response.output
            output_tokens = getattr(response, "output_tokens", None)
            original_tokens = getattr(response, "original_input_tokens", None)
            metrics = build_metrics(
                text,
                compressed,
                original_tokens_override=original_tokens or None,
                compressed_tokens_override=output_tokens if output_tokens is not None else None,
            )
            return {
                "available": True,
                "text": compressed,
                "output_tokens": output_tokens,
                "original_tokens": original_tokens,
                "metrics": metrics,
                "compression_ratio": getattr(response, "compression_ratio", None),
                "tokens_saved": getattr(response, "tokens_saved", None),
                "compression_percentage": getattr(response, "compression_percentage", None),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}
