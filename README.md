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

---

## 🚀 How to Use This Repository
Each lab directory contains:
1. **The original Lab PDF:** Outlining the academic requirements and tasks.
2. **The Source Code:** The Python (and Prolog) scripts used to run the experiments.
3. **`submission.md`:** A beautifully formatted, easy-to-read markdown report answering every prompt, providing real-world examples, and explaining the core concepts in simple terms.

Happy coding!
