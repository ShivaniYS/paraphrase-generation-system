import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import time
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import sent_tokenize

class CustomParaphraseGenerator:
    def __init__(self, model_name: str = "t5-base"):
        """
        Initialize the custom paraphrase generator.
        Uses T5 model which is good for text-to-text tasks.
        """
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def preprocess_text(self, text: str, max_chunk_length: int = 512) -> List[str]:
        """
        Split long paragraphs into manageable chunks while preserving sentence boundaries.
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Create chunks based on token count
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)
            
            if current_length + sentence_length > max_chunk_length and current_chunk:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def paraphrase_chunk(self, text: str, length_ratio: float = 1.0) -> str:
        """
        Paraphrase a single chunk of text.
        """
        # Prepare input with paraphrase prompt
        input_text = f"paraphrase: {text}"
        
        # Tokenize input
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        # Calculate target length based on input
        input_length = len(inputs[0])
        min_length = max(50, int(input_length * 0.8 * length_ratio))
        max_length = int(input_length * 1.5)
        
        # Generate paraphrase
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=5,
                num_return_sequences=1,
                temperature=0.7,
                repetition_penalty=2.0,
                length_penalty=length_ratio,
                early_stopping=True
            )
        
        # Decode output
        paraphrased_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return paraphrased_text
    
    def paraphrase_paragraph(self, paragraph: str, min_length_ratio: float = 0.8) -> str:
        """
        Paraphrase an entire paragraph, ensuring output meets length requirements.
        """
        start_time = time.time()
        
        # Preprocess: split into chunks if needed
        chunks = self.preprocess_text(paragraph)
        
        # Paraphrase each chunk
        paraphrased_chunks = []
        for chunk in chunks:
            paraphrased_chunk = self.paraphrase_chunk(chunk, length_ratio=min_length_ratio)
            paraphrased_chunks.append(paraphrased_chunk)
        
        # Combine chunks
        final_output = " ".join(paraphrased_chunks)
        
        # Ensure minimum length
        input_words = len(paragraph.split())
        output_words = len(final_output.split())
        
        # If output is too short, try again with different parameters
        if output_words < input_words * min_length_ratio:
            print(f"Output too short ({output_words} words vs required {int(input_words*min_length_ratio)}). Adjusting...")
            # Try with longer target length
            final_output = self.paraphrase_chunk(paragraph, length_ratio=1.2)
        
        end_time = time.time()
        self.inference_time = end_time - start_time
        
        return final_output
    
    def batch_paraphrase(self, paragraphs: List[str], min_length_ratio: float = 0.8) -> List[str]:
        """
        Paraphrase multiple paragraphs.
        """
        results = []
        for para in paragraphs:
            result = self.paraphrase_paragraph(para, min_length_ratio)
            results.append(result)
        return results