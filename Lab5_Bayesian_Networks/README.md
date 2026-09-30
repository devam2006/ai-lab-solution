# Lab 5: Bayesian Networks and Autoregressive Language Models

This directory contains the complete solution, source code, test suite, and submission report for **Lab 5: Bayesian Networks and Autoregressive Language Models**.

## 📖 Overview

This laboratory connects fundamental probabilistic graphical models (Bayesian Networks) to the engine driving modern artificial intelligence: **Autoregressive Language Models**.

By factorizing sequence probabilities using the chain rule of probability:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_1, \dots, X_{t-1})$$
we demonstrate that autoregressive text generation is fundamentally **ancestral sampling across a directed acyclic graphical model**.

---

## 🗂 File Structure

- `bn_lab.pdf`: The official laboratory exercise and assignment specification.
- `language_model.py`: Pure Python implementation of First-Order ($P(X_t \mid X_{t-1})$) and Second-Order ($P(X_t \mid X_{t-2}, X_{t-1})$) Bayesian Language Models, supporting both Greedy and Sampling generation modes.
- `test_language_model.py`: Automated `pytest` test suite verifying probabilistic invariants ($\sum_v P(v \mid w) = 1.0$), transition frequencies, determinism, and context disambiguation.
- `submission.md`: The complete academic submission addressing all 14 questions, deliverables, CPT calculations, LLM reflections, and comparative analyses.
- `generated_sentences_first_order.txt`: 20 sampled sentences generated from the First-Order model.
- `generated_sentences_second_order.txt`: 20 sampled sentences generated from the Second-Order model.

---

## ⚡ Quick Start

### 1. Run the Full Experimental Suite
Execute the entire pipeline to view tokenization, CPTs, invariant tests, generation comparisons, and second-order metrics:
```bash
# From workspace root
./venv/bin/python Lab5_Bayesian_Networks/language_model.py

# Or from within this directory
python language_model.py
```

### 2. Run Automated Property Tests
Verify that all mathematical invariants strictly hold:
```bash
./venv/bin/pytest Lab5_Bayesian_Networks/test_language_model.py
```

---

## 🔬 Key Experimental Findings

1. **Greedy Traps:** In the first-order model, greedy ($\arg\max$) generation falls into an inescapable deterministic cycle (`the -> cat -> sat -> on -> the`), repeating infinitely. Sampling, in contrast, explores all valid branches and reaches the `<END>` token.
2. **First-Order Memory Loss:** Because the first-order network forgets history beyond the immediately preceding word, it cannot distinguish `the` at the beginning of a sentence from `the` after a preposition, resulting in ungrammatical phrases like `"sat on the cat"` (34% incoherence rate).
3. **Second-Order Disambiguation:** Conditioning on bigrams ($X_{t-2}, X_{t-1}$) drops semantic incoherence to **0.0%**, perfectly isolating nouns following prepositions (`mat`, `rug`) from subject animals (`cat`, `dog`).
4. **The Curse of Dimensionality:** While the second-order model improves coherence, zero-probability context sparsity explodes from **8.3%** in first-order to **90.3%** in second-order. This fundamental bottleneck reveals why modern LMs use dense neural network weights rather than discrete CPTs.
