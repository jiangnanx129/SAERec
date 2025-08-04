# SAERec: Constructing Interpretable Intent Priors via Sparse Autoencoders for Recommendation

This repository contains the code and resources for reproducing the results of our paper:

> **SAERec: Constructing Interpretable Intent Priors via Sparse Autoencoders for Recommendation**   

## 🔍 Overview

SAERec is a novel intent-aware recommender system that constructs a comprehensive set of **interpretable user intents** from user reviews using **Sparse Autoencoders (SAE)** and **Large Language Models (LLMs)**. These intents are used as semantic priors to guide sequential recommendation via a **multi-branch attention** mechanism.

**Key innovations:**
- Intent extraction is performed on **review embeddings**, not noisy interaction sequences.
- The model uses **LLM-prompting** to identify human-interpretable intents in an unsupervised fashion.
- A **personal-public intent retrieval and injection mechanism** is proposed to enhance personalization and generalization.
- SAERec consistently outperforms state-of-the-art baselines on four benchmark datasets (Amazon Beauty, Toys, Sports, and Yelp).

---
