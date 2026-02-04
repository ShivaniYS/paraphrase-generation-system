# main.py
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.paraphrase import ParaphraseSystem

def main():
    # Sample text for testing
    sample_text = """I am writing to express my enthusiastic interest in the Machine Learning Engineer position at your company. With over five years of experience in developing and deploying machine learning models in production environments, I have developed a strong foundation in both theoretical concepts and practical applications. My expertise includes natural language processing, computer vision, and time-series forecasting, with proficiency in Python, TensorFlow, PyTorch, and cloud platforms like AWS and GCP.

In my previous role at Tech Innovations Inc., I led a team that developed a recommendation system that increased user engagement by 35%. I implemented transformer-based models for text classification and built automated pipelines for model training and deployment. I am particularly drawn to your company's focus on ethical AI and innovative solutions, and I am confident that my skills in model optimization and collaborative problem-solving would contribute significantly to your team.

I have attached my resume for your review and would welcome the opportunity to discuss how my experience aligns with your needs. Thank you for considering my application."""
    
    print("Initializing Paraphrase System...")
    system = ParaphraseSystem(use_llm_api=False)
    
    print(f"\nOriginal Text Length: {len(sample_text.split())} words")
    print("-" * 60)
    
    # Run comparison
    results = system.run_comparison(sample_text, min_length_ratio=0.8)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()