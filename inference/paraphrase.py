import time
from models.cpg_model import CustomParaphraseGenerator
from models.llm_baseline import LLMParaphraser
from evaluation.metrics import ParaphraseEvaluator


class ParaphraseSystem:
    def __init__(self):
        print("Initializing Paraphrase System...")

        self.cpg = CustomParaphraseGenerator()
        self.llm = LLMParaphraser()
        self.evaluator = ParaphraseEvaluator()

        print("System initialized successfully!")

    def run_comparison(self, input_text, min_length_ratio=0.8):

        word_count = len(input_text.split())
        print(f"\nOriginal Text ({word_count} words)")
        print("=" * 60)

        # -------- CPG --------
        print("\nRunning Custom Paraphrase Generator...")
        start = time.time()
        cpg_output = self.cpg.paraphrase_paragraph(input_text, min_length_ratio)
        cpg_time = time.time() - start

        # -------- LLM --------
        print("\nRunning LLM Baseline...")
        start = time.time()
        llm_output = self.llm.paraphrase_paragraph(input_text, min_length_ratio)
        llm_time = time.time() - start

        # -------- Evaluation --------
        cpg_metrics = self.evaluator.evaluate_paraphrase(input_text, cpg_output)
        llm_metrics = self.evaluator.evaluate_paraphrase(input_text, llm_output)

        print("\n" + "=" * 60)
        print("SYSTEM COMPARISON")
        print("=" * 60)

        print(f"BLEU        CPG: {cpg_metrics['bleu']:.4f} | LLM: {llm_metrics['bleu']:.4f}")
        print(f"ROUGE-L     CPG: {cpg_metrics['rougeL']:.4f} | LLM: {llm_metrics['rougeL']:.4f}")
        print(f"BERTScore   CPG: {cpg_metrics['bertscore']:.4f} | LLM: {llm_metrics['bertscore']:.4f}")
        print(f"LengthRatio CPG: {cpg_metrics['length_ratio']:.4f} | LLM: {llm_metrics['length_ratio']:.4f}")
        print(f"Latency     CPG: {cpg_time:.4f}s | LLM: {llm_time:.4f}s")

        print("=" * 60)
