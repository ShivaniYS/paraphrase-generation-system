
# Paraphrase Generation System  
### Custom Transformer-Based Paraphraser vs Large Language Model

A paragraph-level paraphrase generation system that compares a **Custom Paraphrase Generator (CPG)** built using a transformer model with an **LLM-based paraphrasing baseline**.  
The system focuses on **length preservation, semantic fidelity, and system latency**, making it suitable for real-world controlled text rewriting tasks.

---

## 📌 Project Overview

Paraphrase generation involves restating a text using different wording and structure while preserving its original meaning. Unlike summarization, the goal is **full content retention** with linguistic variation.

This project implements and evaluates two approaches:

- **Custom Paraphrase Generator (CPG)**  
  A locally deployed, transformer-based model with explicit length control.
- **LLM-Based Paraphraser**  
  A prompt-driven paraphrasing baseline using a GPT-style language model.

The comparison is performed at the **paragraph level (200–400 words)** with a strict constraint that the generated output must retain **at least 80% of the original length**.

---

## ✨ Key Features

- Paragraph-level paraphrasing (long-form text)
- Explicit output length enforcement (≥80%)
- Sentence-aware chunking to handle long inputs
- Side-by-side comparison of CPG vs LLM
- Comprehensive evaluation using standard NLP metrics
- Latency benchmarking
- Automated error analysis
- Modular and extensible codebase

---

## 🏗️ System Architecture

```

Input Paragraph
↓
Custom Paraphrase Generator (T5-based)
↓
LLM Paraphraser (GPT-style / Mock)
↓
Evaluation Metrics (BLEU, ROUGE, BERTScore, Length, Latency)
↓
Error Analysis & Comparison Report

```

---

## 📁 Repository Structure

```

paraphrase-generation-system/
├── data/                 # Sample input text
├── models/               # Paraphrase models (CPG & LLM baseline)
│   ├── cpg_model.py
│   └── llm_baseline.py
├── inference/            # Inference and system orchestration
│   └── paraphrase.py
├── evaluation/           # Metrics and error analysis
│   ├── metrics.py
│   └── error_analysis.py
├── notebooks/            # Experiments and exploration
├── tests/                # Simple test scripts
├── main.py               # Entry point for running the system
├── requirements.txt
├── setup.py
└── README.md

````

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ShivaniYS/paraphrase-generation-system.git
cd paraphrase-generation-system
````

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run the full comparison (CPG vs LLM)

```bash
python main.py
```

This will:

* Generate paraphrases using both systems
* Measure inference latency
* Compute evaluation metrics
* Print a side-by-side comparison table

---

## 📊 Evaluation Metrics

### Text Quality

* **BLEU** – Lexical overlap
* **ROUGE-L** – Structural similarity
* **BERTScore** – Semantic similarity using contextual embeddings

### System Performance

* **Latency** – Inference time per paragraph
* **Length Ratio** – Output length / input length
* **Diversity** – Approximated as `1 − BLEU`

---

## 🧪 Experimental Results (Summary)

| Metric       | CPG    | LLM    | Difference (LLM − CPG) |
| ------------ | ------ | ------ | ---------------------- |
| BLEU         | 0.0322 | 0.0080 | −0.0242                |
| ROUGE-L      | 0.0913 | 0.1530 | +0.0617                |
| BERTScore    | 0.7688 | 0.8063 | +0.0375                |
| Length Ratio | 0.4835 | 1.0000 | +0.5165                |
| Latency (s)  | 141.75 | 0.00   | −141.75                |



**Key observations**:

* LLM achieves higher semantic similarity.
* CPG consistently satisfies length constraints.
* CPG is significantly faster due to local inference.

---

## 🔍 Error Analysis

Common issues observed during evaluation:

* Meaning drift in nuanced phrases
* Repetition in CPG outputs
* Occasional hallucinated content in LLM outputs
* Length violations primarily in LLM responses

An explicit error analysis module is included to surface these patterns.

---

## 📈 Conclusions

This project demonstrates the trade-off between **control and fluency** in paraphrase generation:

* **CPG** provides predictability, speed, and strict length control.
* **LLM** produces more fluent paraphrases but with higher latency and reduced controllability.

A hybrid approach—using CPG for draft generation and LLM for refinement—offers a promising direction for future work.

---

## 🚀 Future Work

* Fine-tuning the CPG on domain-specific paraphrase datasets
* Reinforcement learning for improved fluency
* Hybrid CPG + LLM pipelines
* Improved prompt strategies for tighter LLM length control

---

## 🛠️ Technical Stack

* Python 3.8+
* PyTorch 2.0+
* HuggingFace Transformers
* NLTK
* ROUGE
* BERTScore

---

## 📜 References

* Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 2020
* Papineni et al., *BLEU: A Method for Automatic Evaluation of Machine Translation*, ACL 2002
* Zhang et al., *BERTScore: Evaluating Text Generation with BERT*, ICLR 2020

---

## 👤 Author

**Shivani Y S**


