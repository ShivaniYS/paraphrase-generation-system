import numpy as np
from typing import Tuple, Dict, List
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch

class ParaphraseEvaluator:
    def __init__(self):
        """Initialize evaluation metrics."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score between reference and candidate.
        """
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        # BLEU with smoothing for short texts
        try:
            score = sentence_bleu([ref_tokens], cand_tokens, 
                                 smoothing_function=self.smoothie)
            return score
        except:
            return 0.0
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        """
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_bertscore(self, reference: str, candidate: str) -> float:
        """
        Calculate BERTScore for semantic similarity.
        """
        try:
            P, R, F1 = bert_score([candidate], [reference], 
                                 lang='en', verbose=False)
            return F1.item()
        except:
            # Fallback if BERTScore fails
            return 0.5
    
    def calculate_length_ratio(self, reference: str, candidate: str) -> float:
        """
        Calculate length ratio: candidate length / reference length.
        """
        ref_words = len(reference.split())
        cand_words = len(candidate.split())
        
        if ref_words == 0:
            return 0.0
        
        return cand_words / ref_words
    
    def evaluate_paraphrase(self, original: str, paraphrase: str) -> Dict[str, float]:
        """
        Comprehensive evaluation of a paraphrase.
        """
        metrics = {}
        
        # BLEU
        metrics['bleu'] = self.calculate_bleu(original, paraphrase)
        
        # ROUGE
        rouge_scores = self.calculate_rouge(original, paraphrase)
        metrics.update(rouge_scores)
        
        # BERTScore
        metrics['bertscore'] = self.calculate_bertscore(original, paraphrase)
        
        # Length ratio
        metrics['length_ratio'] = self.calculate_length_ratio(original, paraphrase)
        
        # Diversity (self-BLEU between original and paraphrase, lower is better)
        metrics['diversity'] = 1 - metrics['bleu']  # Simple diversity measure
        
        return metrics
    
    def compare_systems(self, original: str, cpg_output: str, llm_output: str, 
                       cpg_time: float, llm_time: float) -> Dict:
        """
        Compare CPG vs LLM performance.
        """
        # Evaluate CPG
        cpg_metrics = self.evaluate_paraphrase(original, cpg_output)
        cpg_metrics['latency'] = cpg_time
        
        # Evaluate LLM
        llm_metrics = self.evaluate_paraphrase(original, llm_output)
        llm_metrics['latency'] = llm_time
        
        # Comparison results
        comparison = {
            'cpg': cpg_metrics,
            'llm': llm_metrics,
            'differences': {}
        }
        
        # Calculate differences
        for metric in cpg_metrics:
            if metric in llm_metrics:
                comparison['differences'][metric] = llm_metrics[metric] - cpg_metrics[metric]
        
        return comparison
    
    def print_comparison_table(self, comparison: Dict):
        """
        Print formatted comparison table.
        """
        print("\n" + "="*60)
        print("SYSTEM COMPARISON")
        print("="*60)
        print(f"{'Metric':<15} {'CPG':<12} {'LLM':<12} {'Difference':<12}")
        print("-"*60)
        
        metrics_order = ['bleu', 'rougeL', 'bertscore', 'length_ratio', 'latency']
        metric_names = {
            'bleu': 'BLEU',
            'rougeL': 'ROUGE-L',
            'bertscore': 'BERTScore',
            'length_ratio': 'Length Ratio',
            'latency': 'Latency (s)'
        }
        
        for metric in metrics_order:
            if metric in comparison['cpg']:
                cpg_val = comparison['cpg'][metric]
                llm_val = comparison['llm'][metric]
                diff = comparison['differences'].get(metric, 0)
                
                # Format based on metric type
                if metric == 'latency':
                    print(f"{metric_names[metric]:<15} {cpg_val:.4f}s     {llm_val:.4f}s     {diff:+.4f}s")
                else:
                    print(f"{metric_names[metric]:<15} {cpg_val:.4f}       {llm_val:.4f}       {diff:+.4f}")
        
        print("="*60)
        
        # Winner analysis
        print("\nWINNER ANALYSIS:")
        print("-"*40)
        
        # Determine winners
        winners = {}
        for metric in ['bleu', 'rougeL', 'bertscore']:
            if comparison['cpg'][metric] > comparison['llm'][metric]:
                winners[metric] = 'CPG'
            else:
                winners[metric] = 'LLM'
        
        # Length requirement check
        cpg_meets_length = comparison['cpg']['length_ratio'] >= 0.8
        llm_meets_length = comparison['llm']['length_ratio'] >= 0.8
        
        print(f"Length Requirement (≥80%): CPG: {'✓' if cpg_meets_length else '✗'}, "
              f"LLM: {'✓' if llm_meets_length else '✗'}")
        
        if cpg_meets_length and not llm_meets_length:
            print("✓ CPG meets length requirement, LLM does not")
        elif llm_meets_length and not cpg_meets_length:
            print("✓ LLM meets length requirement, CPG does not")
        elif cpg_meets_length and llm_meets_length:
            print("✓ Both systems meet length requirement")
        else:
            print("✗ Neither system meets length requirement")
        
        # Latency comparison
        latency_ratio = comparison['llm']['latency'] / comparison['cpg']['latency'] if comparison['cpg']['latency'] > 0 else float('inf')
        print(f"\nLatency: CPG is {latency_ratio:.2f}x {'faster' if latency_ratio > 1 else 'slower'} than LLM")