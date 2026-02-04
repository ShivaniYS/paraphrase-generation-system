# test_simple.py
# Run this directly to test the system
import sys
import os

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import
from inference.paraphrase import ParaphraseSystem, load_sample_text


# Initialize and test
system = ParaphraseSystem(use_llm_api=False)

# Use sample text
text = """Artificial intelligence and machine learning are transforming industries worldwide. 
Companies are leveraging AI to automate processes, gain insights from data, and create innovative products. 
The demand for AI professionals has never been higher, with roles ranging from data scientists to ML engineers. 
Successful AI implementation requires both technical expertise and understanding of business contexts."""

print("=" * 70)
print("PARAPHRASE SYSTEM TEST")
print("=" * 70)

results = system.run_comparison(text, min_length_ratio=0.8)

print("\n✓ Test completed successfully!")
print(f"Original length: {len(text.split())} words")
print(f"CPG output length: {len(results['cpg_output'].split())} words")
print(f"LLM output length: {len(results['llm_output'].split())} words")