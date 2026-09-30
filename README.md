# Artificial Intelligence Laboratory Solutions

Welcome to the **AI Labs Solutions** repository! This repository contains comprehensive solutions, source code, and reflection reports for the core Artificial Intelligence undergraduate lab exercises. 

This repository is structured to document both the **AI Engineering** and **AI Science** concepts we learned. Instead of just "coding the answers," these labs focus on using Large Language Models (LLMs) as engineering assistants to write boilerplate code, while we—the human engineers—take full responsibility for designing the environment, specifying the rules, writing tests, and interpreting the output logically.

---

## 📂 Repository Structure

The labs are organized into the following subdirectories:

### [Lab 1: Logical Reasoning for Planning](./Lab1_Logic/)
- **Core Concept**: Logic + Search = Planning.
- **What We Did**: We solved a warehouse puzzle (moving a package from A to C) by strictly defining the logical rules of the world (Preconditions and Effects). We then used an LLM to generate a Breadth-First Search (BFS) engine to explore these rules.
- **Key Takeaway**: We learned how to write strict, formal rules so an AI cannot "hallucinate" physically impossible moves. We also used a rigid logic engine (Prolog) to independently verify the LLM's outputs.

### [Lab 2: Search and A*](./Lab2_Search/)
- **Core Concept**: Informed vs. Uninformed Search.
- **What We Did**: We took a grid map with obstacles and compared two ways of finding a path: a "blind" search (BFS) that checks everywhere, and an "informed" search (A*) that uses a heuristic (a smart guess, like the Manhattan distance) to walk straight toward the goal.
- **Key Takeaway**: We learned that making an algorithm "smarter" depends entirely on giving it good information (heuristics) on where to look next, vastly speeding up the time it takes to find a solution.

### [Lab 3: Goal-Based Agents](./Lab3_Agents/)
- **Core Concept**: Designing Autonomous Agents.
- **What We Did**: We looked at the exact same warehouse maze from Lab 2, but analyzed it from the perspective of an *Agent Architecture*. We defined the environment, the sensors, and the decision-making components.
- **Key Takeaway**: We learned the framework of separating the "Problem Understanding" (the science) from the "Software Implementation" (the engineering), effectively learning how to prompt an LLM strictly with a formal specification rather than just asking it to "write a program."

### [Lab 4: Neural Models](./Lab4_Neural_Models/)
- **Core Concept**: Deep Learning, Backpropagation, and Non-Linearity.
- **What We Did**: We built a tiny neural network in PyTorch to act as a safety sensor (an XOR gate). We experimented with different activation functions (Sigmoid, Tanh, ReLU) to see how gradients flow backward through the network to update the weights.
- **Key Takeaway**: We proved mathematically and experimentally why simple straight lines (linear layers) cannot solve complex logical problems (like XOR), and how "hidden non-linear layers" act as the magic glue that allows neural networks to learn complex representations.

### [Lab 5: Bayesian Networks and Autoregressive Language Models](./Lab5_Bayesian_Networks/)
- **Core Concept**: Bayesian Networks, Chain Rule Factorization, and Autoregressive Generation.
- **What We Did**: We connected probabilistic graphical models directly to modern autoregressive language modeling. We implemented First-Order ($P(X_t \mid X_{t-1})$) and Second-Order ($P(X_t \mid X_{t-2}, X_{t-1})$) Markov models from scratch using pure Python data structures, tested probabilistic normalization invariants ($\sum_v P(v \mid w) = 1.0$), contrasted deterministic greedy decoding against probabilistic sampling, and explored why increasing context resolves semantic ambiguity while exponentially increasing parameter sparsity.
- **Key Takeaway**: We demonstrated that all autoregressive language models (from simple $n$-gram Bayes nets to modern Transformers) are fundamentally performing ancestral sampling over a factorized joint probability distribution.

---

## 🛠️ Environment Setup & Installation

A unified virtual environment can be created and configured for all labs:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install all required dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run the Laboratory Solutions

Each lab can be run directly using Python from the root workspace or within its directory:

```bash
# Run Lab 1: Logical Planning
python Lab1_Logic/planner.py

# Run Lab 2: Search and A*
python Lab2_Search/search_agent.py

# Run Lab 3: Goal-Based Agent
python Lab3_Agents/warehouse_agent.py

# Run Lab 4: Neural Models (XOR & Multi-class)
python Lab4_Neural_Models/neural_xor.py

# Run Lab 5: Bayesian Networks & Autoregressive LM Suite
python Lab5_Bayesian_Networks/language_model.py

# Run Lab 5 Automated Property Tests (Pytest)
pytest Lab5_Bayesian_Networks/test_language_model.py
```

Each lab directory contains:
1. **The original Lab PDF:** Outlining the academic requirements and tasks.
2. **The Source Code:** The Python (and Prolog) scripts used to run the experiments.
3. **`submission.md`:** A beautifully formatted, comprehensive markdown report answering every prompt, providing real-world examples, and explaining the core concepts in rigorous detail.

Happy coding!
