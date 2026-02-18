# 🧠 Paraphrase Generation System

## 📌 Overview

This project implements a **Custom Paraphrase Generator (CPG)** and compares it against a **Large Language Model (LLM) baseline** in terms of:

- Text quality
- Semantic similarity
- Structural preservation
- System latency

The goal is to analyze trade-offs between a fine-tuned transformer model and a large hosted LLM.

---

## 🏗️ System Architecture


---

## 🤖 Models Used

### 🔹 Custom Paraphrase Generator (CPG)
- Fine-tuned T5 model: `Vamsi/T5_Paraphrase_Paws`
- Sentence-level paraphrasing to preserve structure
- Local inference (CPU-based)

### 🔹 LLM Baseline
- OpenAI GPT model (`gpt-4o-mini`)
- API-based hosted inference
- Strong contextual and semantic modeling

---

## 📊 Evaluation Metrics

We evaluate paraphrases using:

- **BLEU** → lexical overlap
- **ROUGE-L** → structural similarity
- **BERTScore** → semantic similarity
- **Length Ratio** → ensures ≥ 80% length
- **Latency** → inference time comparison

---

## 📈 Final Results (Test Passage: 462 words)

| Metric        | CPG (T5) | LLM (GPT) |
|--------------|----------|-----------|
| BLEU         | 0.5825   | 0.2090    |
| ROUGE-L      | 0.8799   | 0.6120    |
| BERTScore    | 0.8825   | 0.8924    |
| Length Ratio | 0.8009   | 0.9957    |
| Latency      | 36.15 s  | 16.80 s   |

---

## 🔍 Analysis

### 🟢 Custom T5 Model
- High lexical and structural preservation (BLEU & ROUGE-L)
- Meets ≥80% length requirement
- Higher latency due to beam search on CPU
- Slightly lower semantic flexibility compared to LLM

### 🔵 GPT-Based LLM
- Higher semantic similarity (BERTScore)
- Better contextual understanding
- Lower latency due to optimized hosted infrastructure
- More flexible rephrasing

---

## 🎯 Key Insights

- Sentence-level paraphrasing aligns better with fine-tuned T5 training distribution.
- Fine-tuned transformer models provide strong structural alignment.
- Hosted LLMs offer better semantic generalization and lower inference latency.
- There is a clear tradeoff between controllable local models and large-scale LLM systems.

---

## 🚀 How to Run

1. Clone repository
2. Install dependencies:

```bash
pip install -r requirements.txt

OPENAI_API_KEY=your_key_here
python main.py




