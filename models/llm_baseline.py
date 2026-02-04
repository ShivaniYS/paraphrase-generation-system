import time
from typing import List, Optional
import openai
from openai import OpenAI
import os

class LLMParaphraser:
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize LLM-based paraphraser.
        Can use OpenAI GPT or other API-based models.
        """
        self.model_name = model_name
        self.client = None
        
        if "gpt" in model_name.lower():
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                # Try environment variable
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                else:
                    print("Warning: No OpenAI API key provided. Using mock responses for testing.")
        
        self.inference_time = 0
        
    def paraphrase_paragraph(self, paragraph: str, min_length_ratio: float = 0.8) -> str:
        """
        Paraphrase using LLM API.
        """
        start_time = time.time()
        
        if not self.client:
            # Mock response for testing
            words = paragraph.split()
            n_words = len(words)
            required_words = int(n_words * min_length_ratio)
            
            # Simple word shuffling as mock (replace with actual API call)
            import random
            shuffled = words.copy()
            random.shuffle(shuffled)
            mock_output = " ".join(shuffled[:max(required_words, len(shuffled))])
            
            end_time = time.time()
            self.inference_time = end_time - start_time
            return mock_output
        
        # Calculate target length
        input_words = len(paragraph.split())
        target_min_words = int(input_words * min_length_ratio)
        
        # Prepare prompt
        prompt = f"""Please paraphrase the following paragraph while maintaining its meaning and style. 
        The output must be at least {target_min_words} words long.

        Original paragraph:
        {paragraph}

        Paraphrased version:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that paraphrases text while preserving meaning, style, and length requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            paraphrased_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            # Fallback to simple method
            paraphrased_text = paragraph  # In practice, implement a better fallback
        
        end_time = time.time()
        self.inference_time = end_time - start_time
        
        return paraphrased_text
    
    def batch_paraphrase(self, paragraphs: List[str], min_length_ratio: float = 0.8) -> List[str]:
        """
        Paraphrase multiple paragraphs.
        """
        results = []
        for para in paragraphs:
            result = self.paraphrase_paragraph(para, min_length_ratio)
            results.append(result)
        return results