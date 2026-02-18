import torch
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize


class CustomParaphraseGenerator:
    def __init__(self, model_name="Vamsi/T5_Paraphrase_Paws"):
        print("USING SENTENCE-LEVEL FINETUNED PARAPHRASE MODEL")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def paraphrase_paragraph(self, paragraph, min_length_ratio=0.8):
        start_time = time.time()

        sentences = sent_tokenize(paragraph)
        paraphrased_sentences = []

        for sentence in sentences:
            input_text = "paraphrase: " + sentence + " </s>"

            encoding = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            paraphrased_sentences.append(decoded)

        final_output = " ".join(paraphrased_sentences)

        self.inference_time = time.time() - start_time
        return final_output

