import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMParaphraser:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in .env file.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def paraphrase_paragraph(self, paragraph: str, min_length_ratio: float = 0.8) -> str:
        start_time = time.time()

        input_words = len(paragraph.split())
        target_min_words = int(input_words * min_length_ratio)

        prompt = f"""
Paraphrase the following paragraph while preserving meaning.
The output must be at least {target_min_words} words.

Paragraph:
{paragraph}
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You paraphrase text while preserving meaning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        output_text = response.choices[0].message.content.strip()

        self.inference_time = time.time() - start_time
        return output_text
