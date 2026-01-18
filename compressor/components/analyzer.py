from tokenc import TokenClient
from dotenv import load_dotenv
import os
from openai import OpenAI
import time
import json
from pathlib import Path

load_dotenv()

class CompressionAnalyzer:
    def __init__(self, tokenc_client: TokenClient, openai_client: OpenAI):
        self.token_client = tokenc_client
        self.openai_client = openai_client

    def analyze_compression(self, prompt: str, context: list[str], aggressiveness: float, log_dir: str = "logs"):
        
        compressed_contexts = []
        compression_times = []
        for ctx in context:
            compressed = self.token_client.compress_input(
                input=ctx, aggressiveness=aggressiveness
            )
            compressed_contexts.append(compressed.output)
            compression_times.append(compressed.compression_time)

        # Prepare prompts
        original_prompt = prompt + "\n\n" + "\n".join(context)
        compressed_prompt = prompt + "\n\n" + "\n".join(compressed_contexts)

        # Call OpenAI client with original and compressed prompts
        def call_openai(prompt) -> str:
            try:
                response = self.openai_client.responses.create(
                    model="gpt-5-mini",
                    input=prompt,
                )
                return response.output_text
            except Exception as e:
                return f"Error: {str(e)}"

        original_response = call_openai(original_prompt)
        compressed_response = call_openai(compressed_prompt)

        # Log results to a file
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"compression_log_{int(time.time())}.json"

        log_data = {
            "original_prompt": original_prompt,
            "compressed_prompt": compressed_prompt,
            "original_response": original_response,
            "compressed_response": compressed_response,
            "compression_time": sum(compression_times),
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        return log_file


def main():
    tokenc_key = os.getenv("TOKEN_COMPANY_API_KEY", None)
    openai_key = os.getenv("OPENAI_API_KEY", None)

    if not tokenc_key:
        raise ValueError("Missing required API keys for Token Company")
    if not openai_key:
        raise ValueError("Missing required API keys for OpenAI")

    tc = TokenClient(api_key=tokenc_key)
    oa = OpenAI(api_key=openai_key)
    analyzer = CompressionAnalyzer(tokenc_client=tc, openai_client=oa)

    prompt = "Summarize the following historical event:"
    text = (
        "The Black Death was a plague pandemic that occurred in Europe from 1346 to 1353. "
        "It was one of the most fatal pandemics in human history; as many as 50 million people perished, perhaps 50% of Europe's 14th-century population. "
        "The disease is caused by the bacterium Yersinia pestis and spread by fleas and through the air. The Black Death had far-reaching population, economic, and cultural impacts."
    )

    log_file = analyzer.analyze_compression(prompt=prompt, context=[text], aggressiveness=0.8)
    print(f"Analysis complete. Results logged to: {log_file}")


if __name__ == "__main__":
    main()
