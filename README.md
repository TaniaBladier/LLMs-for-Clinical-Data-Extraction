# LLMs-for-Data-Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)](https://huggingface.co/docs/transformers/index)

This repository explores the use of Large Language Models (LLMs) from HuggingFace for information extraction from unstructured medical texts. The project focuses on transforming triage notes into machine-readable data by applying several LLM techniques with biomedical NLP workflows.

## 🚀 Key Features

The system transforms triage notes into structured data using several techniques:

* **Fine-tuning of LLMs:** Exploring the Hugging Face Transformers ecosystem for domain-specific model adaptation.
* **Prompt-based Extraction:** Utilizing advanced prompting techniques (Few-shot, Chain-of-Thought) for different extraction tasks.
* **Biomedical NLP:** Integrating traditional techniques such as Named Entity Recognition (NER) to improve extraction accuracy.

## 🏥 Use Case

Given a clinical text (e.g., discharge summary or medical report), the system extracts structured information such as:

* **Medical Conditions** (e.g., "Type 2 Diabetes Mellitus")
* **Symptoms** (e.g., "Acute shortness of breath")
* **Treatments & Medications** (e.g., "Metformin 500mg BID")

## 🎯 Objectives

1.  **Evaluate Effectiveness:** Benchmark the performance of various LLMs on medical information extraction tasks.
2.  **Comparative Analysis:** Compare the trade-offs between **Prompt Engineering** vs. **Parameter-efficient Fine-tuning (PEFT)**.
3.  **Reproducible Pipeline:** Build an end-to-end pipeline from raw, unstructured text to validated structured output (JSON/CSV).
4.  **Clinical Foundation:** Pave the way for downstream clinical NLP applications such as clinical decision support and data normalization.

## 📈 Current Status

This repository started as an experimental sandbox and is currently being progressively structured into a modular and reusable pipeline. 

**ToDos:**
- [ ] Refactor experimental notebooks into modular Python scripts.
- [ ] Add support for various LLM backends (OpenAI, Anthropic, Llama 3 via Ollama/vLLM).
- [ ] Implement evaluation scripts using standard metrics (F1, Precision, Recall).

## 🛠️ Getting Started

*(Self-note: You can add installation and usage instructions here as the repo matures)*

```bash
git clone [https://github.com/your-username/LLMs-for-Data-Extraction.git](https://github.com/your-username/LLMs-for-Data-Extraction.git)
cd LLMs-for-Data-Extraction
pip install -r requirements.txt
