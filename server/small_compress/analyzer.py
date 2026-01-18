from tokenc import TokenClient, CompressResponse
from dotenv import load_dotenv
import os
from openai import OpenAI
import time
import json
from pathlib import Path
from .custom_compressor import CustomCompressor
from .vector_store import VectorStore
from .logger import Logger

load_dotenv()

class CompressionAnalyzer:
    def __init__(self, openai_client: OpenAI, custom_compressor: CustomCompressor, logger: Logger):
        self.openai_client = openai_client
        
        # injected logger (defaults to a new Logger for backward compatibility)
        self.logger = logger
        self.custom_compressor = custom_compressor
        self.epochs = 10
        self.log_dir = "logs"

    def analyze_compression(self, prompt: str, texts: list[str], context: str, aggressiveness: float):
        compressed_contexts = []
        compression_times = []
        for text in texts:
            compressed = self.custom_compressor.compress(text=text, context=context, aggressiveness=aggressiveness)
            
            compressed_contexts.append(compressed.output)
            compression_times.append(compressed.compression_time)

        # Prepare prompts
        original_prompt = prompt + "\n" + "\n".join(context)
        compressed_prompt = prompt + "\n" + "\n".join(compressed_contexts)

        original_response = self.call_openai(original_prompt)
        compressed_response = self.call_openai(compressed_prompt)
        
        self.log_res(original_prompt, compressed_prompt, original_response, compressed_response, compression_times)
        
        print(f"Custom compression completed in {sum(compression_times):.2f} seconds over {len(context)} contexts.")
    
    def call_openai(self, prompt) -> str:
        try:
            response = self.openai_client.responses.create(
                model="gpt-5-mini",
                input=prompt,
            )
            return response.output_text
        except Exception as e:
            return f"Error: {str(e)}"

    def log_res(self, original_prompt: str, compressed_prompt: str, original_response: str, compressed_response: str, compression_times: list[float]):
        
        log_path = Path(self.log_dir)
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
    logger = Logger()

    vector_store = VectorStore(logger=logger)
    custom_compressor = CustomCompressor(
        tokenc_client=tc, 
        vector_store=vector_store, 
        logger=logger,
        similarity_cutoff=0.1,
        chunk_size=3,
    )

    analyzer = CompressionAnalyzer(openai_client=oa, custom_compressor=custom_compressor, logger=logger)

    prompt = "Summarize the following historical event:"
    text = [(
        "The Black Death was a plague pandemic that occurred in Europe from 1346 to 1353. "
        "It was one of the most fatal pandemics in human history; as many as 50 million people perished, perhaps 50% of Europe's 14th-century population. "
        "The disease is caused by the bacterium Yersinia pestis and spread by fleas and through the air. The Black Death had far-reaching population, economic, and cultural impacts."
        "Rohan Choudhury, Guanglei Zhu, Sihan Liu, Koichiro Niinuma, Kris Kitani, László Jeni"
        "Robotics Institute, Carnegie Mellon University"
    )]
    context = "I only care about numbers and statistics related to the Black Death."

    analyzer.analyze_compression(prompt=prompt, context=context, texts=text, aggressiveness=0.8)
    print(f"Analysis complete.")


if __name__ == "__main__":
    main()