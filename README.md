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

