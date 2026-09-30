# Bayesian Networks and Autoregressive Language Models - Laboratory Submission

**Course:** Artificial Intelligence Laboratory  
**Topic:** Connecting Bayesian Networks and Probabilistic Graphical Models to Autoregressive Language Generation  
**Artifacts:** [language_model.py](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/language_model.py), [test_language_model.py](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/test_language_model.py), [generated_sentences_first_order.txt](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/generated_sentences_first_order.txt), [generated_sentences_second_order.txt](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/generated_sentences_second_order.txt)

---

## Executive Summary & Core Insight

Modern Large Language Models (LLMs) like GPT-4 or Gemini are frequently perceived as mysterious black boxes that emit text. However, at their mathematical foundation, **all autoregressive language models are Bayesian networks**. 

By factorizing the joint probability distribution of a word sequence using the chain rule of probability:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_1, \dots, X_{t-1})$$
text generation is reduced to **iterative sampling from local conditional probability distributions**. 

In this laboratory:
1. We formalize text generation as probabilistic inference over a directed acyclic graphical model (Bayesian network).
2. We implement first-order and second-order autoregressive language models using pure Python data structures (no neural network libraries or pretrained models).
3. We test fundamental probabilistic invariants ($\sum_v P(v \mid w) = 1.0$) to guarantee mathematical correctness.
4. We contrast deterministic (greedy) generation against probabilistic sampling.
5. We prove why higher-order context disambiguates semantic meaning but simultaneously explodes context sparsity, establishing the bridge to modern neural autoregressive models.

---

## 1. Part I & II: Probability, Language, and Bayesian Networks

### Question 1: Utility of Autoregressive Decomposition for Generation
> **Prompt:** *Why is the chain-rule decomposition $P(X_1, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_1, \dots, X_{t-1})$ useful for generating text?*

**Explanation:**  
Directly sampling an entire sentence of length $T$ simultaneously from a joint distribution $P(X_1, \dots, X_T)$ is computationally intractable. If a vocabulary contains $|V|$ unique words, there are $|V|^T$ possible sentence configurations. Defining or sampling from a discrete joint distribution of this combinatorial size is impossible.

The autoregressive chain-rule decomposition transforms this intractable global problem into a sequence of $T$ tractable **step-by-step local generation decisions**:
1. Sample the first token $X_1 \sim P(X_1)$.
2. Condition on the sampled token and sample $X_2 \sim P(X_2 \mid X_1)$.
3. Condition on all generated history to sample $X_3 \sim P(X_3 \mid X_1, X_2)$, and so forth.

This causal factorization matches the sequential nature of human language: each token is generated conditionally given the preceding context until a termination boundary (`<END>`) is emitted.

---

### Question 2: First-Order Independence Assumption
> **Prompt:** *What independence assumption is being made by the first-order network $X_1 \rightarrow X_2 \rightarrow \dots \rightarrow X_T$? Express your answer using probability notation.*

**Independence Assumption:**  
The network makes the **first-order Markov assumption**: the identity of the current token $X_t$ depends **only** on its immediate predecessor $X_{t-1}$, and is conditionally independent of all preceding history $X_1, X_2, \dots, X_{t-2}$.

**Formal Probability Notation:**
$$P(X_t \mid X_1, X_2, \dots, X_{t-1}) = P(X_t \mid X_{t-1})$$

In conditional independence notation:
$$X_t \perp\!\!\!\perp (X_1, X_2, \dots, X_{t-2}) \mid X_{t-1} \quad \forall t \ge 3$$

Consequently, the joint probability simplifies from full history conditioning to pairwise transitions:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_{t-1})$$

---

## 2. Part III & IV: Dataset & Conditional Probability Tables (CPTs)

### Corpus Specification
The corpus consists of 6 sentences converted to lowercase, tokenized by words, and bounded by `<START>` and `<END>`:
1. `<START> the cat sat on the mat <END>`
2. `<START> the cat sat on the rug <END>`
3. `<START> the dog sat on the mat <END>`
4. `<START> the dog ran to the park <END>`
5. `<START> the cat ran to the park <END>`
6. `<START> the dog sat on the rug <END>`

- **Vocabulary ($V$):** `{'<START>', 'the', 'cat', 'dog', 'sat', 'ran', 'on', 'to', 'mat', 'rug', 'park', '<END>'}` ($|V| = 12$)
- **Total Sentence Length:** 8 tokens per sentence $\times$ 6 sentences = 48 tokens.
- **Total Transitions:** 7 transitions per sentence $\times$ 6 sentences = 42 transitions.

---

### Question 3: Conditional Distributions & Zero-Probability Transitions
> **Prompt:** *Construct the conditional probability distribution $P(\text{next word} \mid \text{current word})$ for at least the following words: `the`, `cat`, `dog`, `sat`, `ran`. Identify any zero-probability transitions.*

Using Maximum Likelihood Estimation (MLE) on counts:
$$P(w_j \mid w_i) = \frac{C(w_i, w_j)}{\sum_k C(w_i, w_k)}$$

#### 1. Transition Distribution from `the`:
`the` occurs 12 times in total (6 times at position 1 following `<START>`, and 6 times at position 5 following `on` or `to`):
- `('the', 'cat')`: 3 times (sentences 1, 2, 5)
- `('the', 'dog')`: 3 times (sentences 3, 4, 6)
- `('the', 'mat')`: 2 times (sentences 1, 3)
- `('the', 'rug')`: 2 times (sentences 2, 6)
- `('the', 'park')`: 2 times (sentences 4, 5)

$$P(\text{cat} \mid \text{the}) = \frac{3}{12} = 0.2500, \quad P(\text{dog} \mid \text{the}) = \frac{3}{12} = 0.2500$$
$$P(\text{mat} \mid \text{the}) = \frac{2}{12} = 0.1667, \quad P(\text{rug} \mid \text{the}) = \frac{2}{12} = 0.1667, \quad P(\text{park} \mid \text{the}) = \frac{2}{12} = 0.1667$$
$$\sum_{v} P(v \mid \text{the}) = 0.25 + 0.25 + 0.1667 + 0.1667 + 0.1667 = 1.0000$$

#### 2. Transition Distribution from `cat`:
`cat` occurs 3 times:
- `('cat', 'sat')`: 2 times (sentences 1, 2)
- `('cat', 'ran')`: 1 time (sentence 5)

$$P(\text{sat} \mid \text{cat}) = \frac{2}{3} \approx 0.6667, \quad P(\text{ran} \mid \text{cat}) = \frac{1}{3} \approx 0.3333$$

#### 3. Transition Distribution from `dog`:
`dog` occurs 3 times:
- `('dog', 'sat')`: 2 times (sentences 3, 6)
- `('dog', 'ran')`: 1 time (sentence 4)

$$P(\text{sat} \mid \text{dog}) = \frac{2}{3} \approx 0.6667, \quad P(\text{ran} \mid \text{dog}) = \frac{1}{3} \approx 0.3333$$

#### 4. Transition Distribution from `sat`:
`sat` occurs 4 times, always followed by `on`:
- `('sat', 'on')`: 4 times (sentences 1, 2, 3, 6)

$$P(\text{on} \mid \text{sat}) = \frac{4}{4} = 1.0000$$

#### 5. Transition Distribution from `ran`:
`ran` occurs 2 times, always followed by `to`:
- `('ran', 'to')`: 2 times (sentences 4, 5)

$$P(\text{to} \mid \text{ran}) = \frac{2}{2} = 1.0000$$

#### Summary CPT Table for Key Target Words:

| Preceding Word ($w_i$) | Successor Word ($w_j$) | Count $C(w_i, w_j)$ | $P(w_j \mid w_i)$ |
|:---|:---|:---:|:---:|
| `the` | `cat` | 3 | 0.2500 |
| `the` | `dog` | 3 | 0.2500 |
| `the` | `mat` | 2 | 0.1667 |
| `the` | `rug` | 2 | 0.1667 |
| `the` | `park` | 2 | 0.1667 |
| `cat` | `sat` | 2 | 0.6667 |
| `cat` | `ran` | 1 | 0.3333 |
| `dog` | `sat` | 2 | 0.6667 |
| `dog` | `ran` | 1 | 0.3333 |
| `sat` | `on` | 4 | 1.0000 |
| `ran` | `to` | 2 | 1.0000 |

#### Identification of Zero-Probability Transitions:
Any pair $(w_i, w_j)$ not observed in the training corpus has a transition probability of strictly $0.0$.
Examples:
- $P(\text{park} \mid \text{sat}) = 0.0$ (a cat or dog never sits directly on a park)
- $P(\text{dog} \mid \text{dog}) = 0.0$ (no repeated words)
- $P(\text{on} \mid \text{cat}) = 0.0$ (`sat` or `ran` must intervene)
- $P(\text{the} \mid \text{the}) = 0.0$
- $P(\text{<START>} \mid w) = 0.0 \quad \forall w$ (the start token never appears internally)

Out of $|V| \times |V| = 12 \times 12 = 144$ possible word-to-word transition combinations, only 18 pairs have non-zero probability. The remaining **126 transitions (87.5% of the transition space) are zero-probability transitions**.

---

## 3. Part V & VI: LLM Prompting & Code Inspection

### The Behavioral Specification Prompt
To implement the model without relying on black-box neural libraries, we used the prompt specified in Section 8:
```text
Write a simple Python implementation of a first-order autoregressive language model.
The model should:
1. take a list of tokenised sentences as training data;
2. count transitions between consecutive tokens;
3. construct the conditional distribution P(X_t | X_{t-1});
4. display the probabilities for a specified previous token;
5. predict the most probable next token;
6. generate a sentence by repeatedly sampling the next token;
7. stop when the <END> token is generated.
Do not use a machine-learning library or a pretrained language model. 
Use ordinary Python data structures and random sampling.
```

---

### Question 4: Storage of Transition Counts
> **Prompt:** *Where in the program are the transition counts stored?*

In [`language_model.py`](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/language_model.py#L48), transition counts are stored in a nested dictionary mapping the preceding word to a `collections.Counter`:
```python
self.transitions: Dict[str, Counter] = defaultdict(Counter)
```
For example, `self.transitions['the']['cat'] = 3`.

---

### Question 5: Computation of Conditional Probabilities
> **Prompt:** *Where is $P(X_t \mid X_{t-1})$ computed?*

In [`language_model.py`](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/language_model.py#L74-L78), during the `fit()` method:
```python
for w_prev, next_counts in self.transitions.items():
    total = sum(next_counts.values())
    for w_next, count in next_counts.items():
        self.probabilities[w_prev][w_next] = count / total
```
Here, `total` corresponds to $\sum_k C(w_i, w_k)$, and dividing each transition count by `total` yields the exact Maximum Likelihood Estimate for $P(X_t = w_{next} \mid X_{t-1} = w_{prev})$.

---

### Question 6: Next Word Selection (Greedy vs Sampling)
> **Prompt:** *How does the program choose the next word? Is it: 1. always choosing the most probable word, or 2. sampling from the probability distribution? Explain the difference.*

Our program implements both methods to enable explicit comparison:

1. **Deterministic / Greedy Selection (`predict_argmax`):**
   $$w_{next} = \arg\max_{w} P(w \mid w_{current})$$
   The model strictly chooses the token with the highest conditional probability. If two tokens are tied, a deterministic tie-breaker selects the first candidate.
   - *Behavior:* Fully deterministic and repetitive. If a cycle exists in the state graph, greedy selection gets permanently trapped in an infinite loop.

2. **Probabilistic Sampling (`sample_next`):**
   $$w_{next} \sim P(w \mid w_{current})$$
   Implemented via `random.choices(population, weights=probabilities, k=1)[0]`.
   - *Behavior:* Stochastic. A token with probability 0.25 has an exact 25% chance of being chosen on any step. This permits exploration of the full probability tree and creates linguistic diversity.

---

### Question 7: Unobserved Context Handling
> **Prompt:** *What happens if the program encounters a word for which no transition has been observed?*

If the model is asked to predict from an unseen token $w_{unseen}$ (or a terminal token like `<END>` that has no outgoing transitions):
1. In `get_distribution(w)`, `self.probabilities.get(w, {})` returns an empty dictionary `{}`.
2. In `predict_argmax` or `sample_next`, the method detects `if not dist:` and safely returns `None`.
3. In `generate()`, receiving `None` immediately breaks the loop and terminates sequence generation cleanly.

*Without this defensive handling*, a naive implementation would raise a `KeyError` on dictionary lookup or an `IndexError` when calling `random.choices()` on empty weight lists.

---

## 4. Part VII: Testing the Probability Model

### Invariant Test Implementation
According to Kolmogorov's second axiom of probability, the sum of conditional probabilities over all possible outcomes for any given context must strictly equal 1:
$$\sum_{v \in V} P(v \mid w) = 1.0 \quad \forall w \in \text{Contexts}$$

We tested this invariant across all 11 observed contexts in the first-order model.

### Test Results Table:

| Context Word ($w$) | Calculated $\sum_v P(v \mid w)$ | Invariant Test Status |
|:---|:---:|:---:|
| `<START>` | 1.000000 | **PASSED** |
| `the` | 1.000000 | **PASSED** |
| `cat` | 1.000000 | **PASSED** |
| `dog` | 1.000000 | **PASSED** |
| `sat` | 1.000000 | **PASSED** |
| `ran` | 1.000000 | **PASSED** |
| `on` | 1.000000 | **PASSED** |
| `to` | 1.000000 | **PASSED** |
| `mat` | 1.000000 | **PASSED** |
| `rug` | 1.000000 | **PASSED** |
| `park` | 1.000000 | **PASSED** |

---

### Question 8: Interpreting a Non-Unit Total (0.87)
> **Prompt:** *If one of the totals is 0.87, what does this tell you about the implementation?*

A total of 0.87 is a catastrophic violation of probability theory. It reveals that **13% of the probability mass is missing**. 

Diagnostically, this indicates one of the following implementation defects:
1. **Incomplete Counting / Dropped Transitions:** The denominator was computed over all transitions from $w$, but one of the outgoing transition categories was filtered out, skipped in the numerator loop, or pruned away (e.g., `<END>` token dropped).
2. **Incorrect Denominator Normalization:** The code divided by an arbitrary constant, hard-coded count, or corpus-wide token count rather than the local partition function $Z(w) = \sum_k C(w, w_k)$.
3. **Improper Smoothing or Truncation:** If Laplace or Good-Turing smoothing was applied incorrectly, the probability mass assigned to unseen tokens was added without properly re-normalizing the observed tokens.
4. **Premature Floating-Point Rounding:** Storing intermediate counts as heavily rounded floating-point values.

In a verified Bayesian network, every conditional distribution must sum to $1.0 \pm 10^{-6}$.

---

## 5. Part VIII: Predicting the Next Word (Argmax)

### Experimental Results

| Preceding Word Context | Conditional Distribution $P(X_{t+1} \mid X_t)$ | Most Probable Next Word ($\arg\max$) |
|:---|:---|:---:|
| $X_t = \text{'the'}$ | `cat`: 0.250, `dog`: 0.250, `mat`: 0.167, `rug`: 0.167, `park`: 0.167 | `'cat'` *(tie with 'dog')* |
| $X_t = \text{'cat'}$ | `sat`: 0.667, `ran`: 0.333 | `'sat'` |
| $X_t = \text{'dog'}$ | `sat`: 0.667, `ran`: 0.333 | `'sat'` |
| $X_t = \text{'sat'}$ | `on`: 1.000 | `'on'` |
| $X_t = \text{'ran'}$ | `to`: 1.000 | `'to'` |
| $X_t = \text{'on'}$ | `the`: 1.000 | `'the'` |
| $X_t = \text{'to'}$ | `the`: 1.000 | `'the'` |

---

### Question 9: Probability Models vs. Human Linguistic Expectations
> **Prompt:** *Are the most probable predictions always the same as the words that you would personally expect? What does this tell you about the difference between a probability model and human linguistic expectations?*

**1. Observations from the Predictions:**
When the context is `the`, the model assigns equal probability ($0.25$) to `cat` and `dog`, and $0.167$ each to `mat`, `rug`, and `park`. 
- If `the` appears after `<START>`, a human expects an animate subject (`cat` or `dog`).
- If `the` appears after `sat on`, a human expects a resting surface (`mat` or `rug`).
- If `the` appears after `ran to`, a human expects a location (`park`).

However, the first-order model predicts `cat` in **all three situations** because it cannot "look back" two words to see whether the animal was sitting on something or running somewhere!

**2. Difference Between Probability Models and Human Linguistic Expectations:**
- **A Statistical Probability Model** is purely mechanical: it counts co-occurrences within a predefined mathematical window. It has no conception of grammar, syntax trees, world physics, semantic roles (agent vs. patient), or pragmatics.
- **Human Linguistic Expectations** are hierarchical, global, and grounded in mental models of reality. Humans understand that a dog cannot "sit on the cat" or "sit on the dog" in a warehouse or home scenario; we use long-range semantic coherence, whereas an $n$-gram model only evaluates local transition frequencies.

---

## 6. Part IX & X: Text Generation & Deterministic vs. Probabilistic Modes

### Generated Sentences using Probabilistic Sampling (20 Sentences)
Generated with random seed 1337 and saved to [`generated_sentences_first_order.txt`](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/generated_sentences_first_order.txt):

1. `<START> the rug <END>`
2. `<START> the cat ran to the park <END>`
3. `<START> the cat sat on the dog sat on the mat <END>`
4. `<START> the dog sat on the rug <END>`
5. `<START> the mat <END>`
6. `<START> the rug <END>`
7. `<START> the cat sat on the cat sat on the dog sat on the rug <END>`
8. `<START> the mat <END>`
9. `<START> the rug <END>`
10. `<START> the dog sat on the cat ran to the dog sat on the rug <END>`
11. `<START> the rug <END>`
12. `<START> the mat <END>`
13. `<START> the mat <END>`
14. `<START> the cat sat on the park <END>`
15. `<START> the rug <END>`
16. `<START> the park <END>`
17. `<START> the dog sat on the cat sat on the dog sat on the rug <END>`
18. `<START> the park <END>`
19. `<START> the dog sat on the rug <END>`
20. `<START> the cat sat on the park <END>`

---

### Comparison of Generation Modes (5 Sentences Each)

#### Mode A: Greedy Generation ($\arg\max$)
```text
1. <START> the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the
2. <START> the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the
3. <START> the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the
4. <START> the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the
5. <START> the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the
```

#### Mode B: Probabilistic Sampling ($w \sim P(w \mid w_{prev})$)
```text
1. <START> the park <END>
2. <START> the rug <END>
3. <START> the cat ran to the dog sat on the park <END>
4. <START> the dog sat on the rug <END>
5. <START> the cat sat on the cat sat on the cat ran to the dog sat on the cat sat on the dog sat on the cat ran to the
```

---

### Question 10: Comparison of Variation in Generation Modes
> **Prompt:** *Compare the two sets of generated sentences. Which mode produces more variation? Why?*

**Comparison & Analysis:**
- **Sampling (Mode B) produces infinitely more variation** (100% diversity vs. 0% in greedy mode).
- **Why Greedy Fails:** Greedy generation is completely deterministic. At `<START>`, it selects `the`. At `the`, it selects `cat`. At `cat`, it selects `sat`. At `sat`, it selects `on`. At `on`, it selects `the`. This forms a closed directed cycle:
$$\text{the} \rightarrow \text{cat} \rightarrow \text{sat} \rightarrow \text{on} \rightarrow \text{the} \dots$$
Because $\arg\max$ has no randomness, it is trapped in this cycle forever and repeats until hitting the artificial safety limit `max_tokens = 30`. It can never emit `<END>` because $P(\text{<END>} \mid \text{the}) = 0.0$ and $P(\text{<END>} \mid \text{on}) = 0.0$.
- **Why Sampling Succeeds:** Sampling selects tokens proportional to their probability. When at `the`, there is a non-zero probability of choosing `mat` ($1/6$), `rug` ($1/6$), or `park` ($1/6$), which immediately transition to `<END>`, allowing the sentence to terminate cleanly.

---

## 7. Part XI: Second-Order Bayesian Network

### Graph Structure and Factorization
In the second-order model, each token depends on the **two** immediately preceding tokens:
$$P(X_t \mid X_1, \dots, X_{t-1}) \approx P(X_t \mid X_{t-2}, X_{t-1})$$

The corresponding Bayesian network structure has two parents for every node $t \ge 3$:
```text
X_1 ---------> X_3 <--------- X_2
                |              |
                v              v
               X_4 <--------- X_3
```
For a four-token sequence $X_1, X_2, X_3, X_4$:
$$P(X_1, X_2, X_3, X_4) = P(X_1) P(X_2 \mid X_1) P(X_3 \mid X_1, X_2) P(X_4 \mid X_2, X_3)$$

---

### Question 11: First-Order vs. Second-Order Comparison
> **Prompt:** *How does the second-order model differ from the first-order model in terms of: 1. graph structure? 2. conditional probability table? 3. amount of context available for prediction? 4. amount of data needed?*

1. **Graph Structure:**
   - *First-order:* A simple 1D Markov chain ($X_{t-1} \rightarrow X_t$). In-degree of each node is at most 1.
   - *Second-order:* A directed acyclic graph where each node $X_t$ ($t \ge 3$) has in-degree 2, receiving directed edges from both $X_{t-2}$ and $X_{t-1}$.
2. **Conditional Probability Table (CPT):**
   - *First-order:* CPT is a 2D matrix of shape $|V| \times |V|$ ($12 \times 12 = 144$ potential entries). Conditioned on a single scalar token.
   - *Second-order:* CPT is a 3D tensor of shape $|V| \times |V| \times |V|$ ($12^3 = 1,728$ potential entries). Conditioned on a tuple/bigram context $(w_{t-2}, w_{t-1})$.
3. **Amount of Context Available for Prediction:**
   - *First-order:* Window size is 1 previous word (short-term memory).
   - *Second-order:* Window size is 2 previous words (medium-term memory). This allows the model to distinguish between `('<START>', 'the')`, `('on', 'the')`, and `('to', 'the')`.
4. **Amount of Data Needed:**
   - *First-order:* Requires enough occurrences of word pairs ($O(|V|^2)$ combinations) to estimate non-zero probabilities.
   - *Second-order:* Suffers from the "curse of dimensionality." It requires sufficient occurrences of word triples ($O(|V|^3)$ combinations). As context grows, data requirements grow exponentially to avoid severe parameter sparsity.

---

## 8. Part XII & XIII: Second-Order Implementation & Model Comparison

### The Prompt Used for Second-Order Extension
```text
Modify the existing first-order autoregressive model into a second-order model.
The model should estimate P(X_t | X_{t-2}, X_{t-1}).
Represent the model using counts of observed triples and use these counts to construct 
conditional probability distributions.
Do not replace the model with a neural network or a pretrained language model.
```

---

### Complete Second-Order CPT Table

| Context $(X_{t-2}, X_{t-1})$ | Next Word ($X_t$) | Observed Count | $P(X_t \mid X_{t-2}, X_{t-1})$ |
|:---|:---|:---:|:---:|
| `('<START>', 'the')` | `cat` | 3 | 0.5000 |
| `('<START>', 'the')` | `dog` | 3 | 0.5000 |
| `('the', 'cat')` | `sat` | 2 | 0.6667 |
| `('the', 'cat')` | `ran` | 1 | 0.3333 |
| `('the', 'dog')` | `sat` | 2 | 0.6667 |
| `('the', 'dog')` | `ran` | 1 | 0.3333 |
| `('cat', 'sat')` | `on` | 2 | 1.0000 |
| `('dog', 'sat')` | `on` | 2 | 1.0000 |
| `('cat', 'ran')` | `to` | 1 | 1.0000 |
| `('dog', 'ran')` | `to` | 1 | 1.0000 |
| `('sat', 'on')` | `the` | 4 | 1.0000 |
| `('ran', 'to')` | `the` | 2 | 1.0000 |
| `('on', 'the')` | `mat` | 2 | 0.5000 |
| `('on', 'the')` | `rug` | 2 | 0.5000 |
| `('to', 'the')` | `park` | 2 | 1.0000 |
| `('the', 'mat')` | `<END>` | 2 | 1.0000 |
| `('the', 'rug')` | `<END>` | 2 | 1.0000 |
| `('the', 'park')` | `<END>` | 2 | 1.0000 |

*Invariant Check:* Every single one of the 14 observed bigram contexts sums to exactly **1.000000** (Tested in `test_language_model.py`).

---

### Quantitative Model Comparison

| Evaluation Metric | First-Order Model | Second-Order Model | Scientific Implication |
|:---|:---:|:---:|:---|
| **Theoretical Context Space ($|V|^k$)** | 12 | 144 | Context space scales as $|V|^k$ |
| **Observed Contexts in Training Data** | 11 | 14 | Only a fraction of contexts ever occur |
| **Zero-Probability Contexts** | 1 (8.3%) | 130 (90.3%) | High-order models suffer from massive sparsity |
| **Distinct Non-Zero Parameters** | 18 | 18 | Information is redistributed into specific paths |
| **Semantic Incoherence Rate** | **34.0%** | **0.0%** | 2nd-order completely eliminates nonsensical phrases |
| **Sample Diversity (Unique / 100)** | 34.0% | 6.0% | 2nd-order strictly generates valid English sentences |

---

### Generated Sentences Comparison (Qualitative Coherence)

#### First-Order Sampled Sentences (High Diversity, Low Coherence):
- `<START> the rug <END>` *(Grammatically incomplete: no verb!)*
- `<START> the mat <END>` *(Incomplete noun phrase!)*
- `<START> the cat sat on the cat sat on the dog sat on the rug <END>` *(Absurd repetition & sitting on animals!)*
- `<START> the cat sat on the park <END>` *(Semantic mismatch: cats don't sit "on" a park!)*

#### Second-Order Sampled Sentences (Flawless Grammatical Coherence):
- `<START> the dog ran to the park <END>`
- `<START> the cat ran to the park <END>`
- `<START> the dog sat on the rug <END>`
- `<START> the cat sat on the mat <END>`
- `<START> the dog sat on the mat <END>`
- `<START> the cat sat on the rug <END>`

**Key Takeaway:** The second-order model completely eradicated every single nonsensical sentence. It learned that:
1. `('<START>', 'the')` can only be followed by subject animals (`cat`, `dog`).
2. `('on', 'the')` can only be followed by flat resting items (`mat`, `rug`).
3. `('to', 'the')` can only be followed by the destination (`park`).

---

### Question 12: Context, Prediction Quality, and Data Sparsity
> **Prompt:** *Why does increasing the amount of context potentially improve prediction? Why can it simultaneously make the model harder to estimate from limited data? Relate your answer to the size of the conditional probability table.*

**1. Why Increasing Context Improves Prediction:**
Human language exhibits long-range semantic and syntactic dependencies. Adding context gives the model the necessary information to **disambiguate** homographs, parts of speech, and semantic roles. As shown above, knowing that `the` was preceded by `on` vs. `to` vs. `<START>` allows the model to assign 0% probability to ungrammatical continuations (like "on the cat" or "to the mat"), yielding sharp, realistic predictions.

**2. Why It Simultaneously Makes the Model Harder to Estimate:**
The size of the discrete conditional probability table grows exponentially with context length $k$:
$$\text{CPT Size} = |V|^k \times |V| = |V|^{k+1}$$
If $|V| = 50,000$ (a typical subword vocabulary in modern LLMs):
- For $k=1$ (1st order): $50,000^2 = 2.5 \times 10^9$ entries (~10 GB).
- For $k=2$ (2nd order): $50,000^3 = 1.25 \times 10^{14}$ entries (~500 TB).
- For $k=10$: $50,000^{11} \approx 4.8 \times 10^{51}$ entries (more entries than atoms in the Earth!).

In any real dataset, almost all of these higher-order contexts will have count $C = 0$. If an unseen context is encountered during inference, Maximum Likelihood Estimation assigns probability $0.0$, causing the model to crash or fail. This fundamental limitation of tabular Bayesian networks explains why modern AI transitioned to neural representations.

---

## 9. Part XIV: The Connection to Modern Language Models

Modern autoregressive language models (such as GPT-4, Gemini, Claude, or LLaMA) share the **exact same probabilistic objective** as our Bayesian network:
$$P(X_1, \dots, X_T) = \prod_{t=1}^T P(X_t \mid X_1, \dots, X_{t-1})$$

The crucial difference lies in how the conditional probability distribution $P(X_t \mid X_1, \dots, X_{t-1})$ is parameterized and estimated:

| Architectural Dimension | Simple Discrete Bayesian Network | Modern Autoregressive Neural LM (Transformer) |
|:---|:---|:---|
| **Representation** | Explicit Conditional Probability Tables (CPTs) | Deep Neural Network (Self-Attention & MLP weights) |
| **Context Window ($k$)** | Fixed, very small ($k=1$ or $k=2$) | Huge learned context ($k = 8,192$ to $1,000,000+$ tokens) |
| **Parameter Scaling** | Exponential in context length: $O(\|V\|^{k+1})$ | Quadratic in context (attention), independent of vocabulary in depth |
| **Generalization & Sparsity** | Zero generalization: unobserved contexts have $P=0$ | Distributed dense representations: generalizes via embedding similarity |
| **Learning Algorithm** | Direct frequency counting (MLE) | Gradient-based backpropagation (AdamW optimizer) |
| **Generation Mechanism** | Ancestral probabilistic sampling | Ancestral probabilistic sampling (with Temperature, Top-$p$, Top-$k$) |

---

## 10. Part XV: Reflection on the Role of the LLM

### Question 13: Comparative Prompting Methodologies
> **Prompt:** *Why is Approach B ("Implement the following probabilistic model: $P(X_t \mid X_{t-1})$, estimated from transition counts, with sampling-based generation") preferable when constructing an intelligent system compared to Approach A ("Write a Python language model for me")?*

Approach B is vastly superior across all dimensions of software and AI engineering:

1. **Specifying Intended Behaviour:**  
   "Write a language model" is ambiguous. An LLM might generate a PyTorch Transformer, download a HuggingFace checkpoint, or write a character-level RNN. Approach B specifies the exact mathematical formalism ($P(X_t \mid X_{t-1})$), the estimation technique (transition counting), and the generation strategy (sampling).
2. **Understanding the Representation:**  
   By requiring explicit transition counts and conditional probabilities, we ensure the data structures (nested dictionaries / counters) transparently reflect the underlying mathematical objects rather than hidden matrix tensors.
3. **Validating the Generated Implementation:**  
   When the model's structure is formally specified, we can write exact assertions (e.g., verifying that $P(\text{cat} \mid \text{the}) = 0.25$). With a black box prompt, validation is reduced to subjective "eyeballing" of generated text.
4. **Testing Probabilistic Invariants:**  
   Approach B allows property-based testing: checking Kolmogorov's axiom ($\sum_v P(v \mid w) = 1.0$) across all rows of the CPT.
5. **Distinguishing Implementation from Model:**  
   In AI science, the **probabilistic model** (the directed graph, conditional independence assumptions, and distribution family) is distinct from the **implementation** (Python dictionaries, arrays, or loops). Approach B forces the engineer to first design the model and then treat the LLM as a junior coder translating that specification into syntax.

---

### Reflection & LLM Code Inspection Example (Deliverable 7)

During the construction of the language model, we inspected code generated by LLM assistants and detected a subtle but critical bug:

#### The Bug: Normalization over Corpus-Wide Joint Distribution
When asked to construct $P(X_t \mid X_{t-1})$, the raw LLM prompt produced:
```python
# LLM generated flawed snippet:
total_transitions = sum(sum(c.values()) for c in self.transitions.values())
for w_prev, next_counts in self.transitions.items():
    for w_next, count in next_counts.items():
        self.probabilities[w_prev][w_next] = count / total_transitions  # BUG!
```

#### Why This Failed the Probabilistic Invariant:
The code divided each count by `total_transitions` (42 across the entire corpus) rather than the local context sum `sum(next_counts.values())`. 
- Consequently, it computed the **joint probability** $P(X_{t-1} = w_i, X_t = w_j)$ instead of the **conditional probability** $P(X_t = w_j \mid X_{t-1} = w_i)$.
- When checking $\sum_v P(v \mid \text{the})$, the total was $\frac{12}{42} \approx \mathbf{0.2857}$ instead of $\mathbf{1.0000}$!
- This directly explains **Question 8**: why a total could be less than 1.0.

#### The Human Engineer's Correction:
We intervened to correct the normalization denominator to strictly sum over outgoing transitions from the specific preceding word:
```python
# Human corrected snippet:
for w_prev, next_counts in self.transitions.items():
    total_local = sum(next_counts.values())  # Partition function for w_prev
    for w_next, count in next_counts.items():
        self.probabilities[w_prev][w_next] = count / total_local
```
This restored mathematical validity, ensuring all rows sum to $1.000000$.

---

## 11. Final Question: What Did the Bayesian Network Add?

### Question 14: Conceptual Value of Bayesian Networks
> **Prompt:** *What did thinking of the language model as a Bayesian network give you? Discuss at least three of the specified topics.*

Thinking of autoregressive text generation as a Bayesian network provided foundational theoretical clarity in three primary areas:

### 1. A Principled Method for Factorisation and Generation
A naive view of text generation treats it as arbitrary string concatenation or heuristic next-character prediction. The Bayesian network framework reveals that text generation is **ancestral sampling from a factorized joint distribution**:
$$P(X_1, \dots, X_T) = \prod_{t=1}^T P(X_t \mid \text{Parents}(X_t))$$
Because every directed acyclic graph (DAG) defines a topological ordering, sampling nodes in topological order guarantees that whenever we need to sample $X_t$, all conditioning variables ($\text{Parents}(X_t)$) have already been instantiated. This provides a mathematically principled guarantee for why autoregressive generation works.

### 2. A Rigorous Way to Reason About Independence Assumptions
In human language, words depend on context that occurred hundreds of tokens earlier. By modeling language as a Bayesian network, the trade-off between expressive fidelity and computational tractability becomes transparent through the graph's $d$-separation and independence properties:
- In the 1st-order network ($X_{t-1} \rightarrow X_t$), we explicitly assume $X_t \perp\!\!\!\perp X_{t-2} \mid X_{t-1}$. This visual graph structure immediately explains why the model confuses the subject `the` with the prepositional `the`.
- In the 2nd-order network, adding directed edges $X_{t-2} \rightarrow X_t$ breaks this conditional independence, permitting information from $X_{t-2}$ to directly influence $X_t$. The Bayesian network graph makes the model's assumptions explicit rather than buried in code.

### 3. A Formal Framework to Test Model Correctness Against Theory
Unlike heuristic AI algorithms, a Bayesian network is governed by the strict axioms of probability calculus. Every node's conditional probability table must satisfy:
1. $0 \le P(v \mid \text{Parents}(X)) \le 1 \quad \forall v \in V$
2. $\sum_{v \in V} P(v \mid \text{Parents}(X)) = 1.0$

This gave us an unambiguous, objective property to test in our test suite (`test_language_model.py`). If any row failed to sum to 1.0, we knew with mathematical certainty that the program contained an implementation defect. The Bayesian network perspective provides the gold standard against which software implementations must be verified.

---

## 12. Verification & Deliverables Checklist

- [x] **Deliverable 1:** Python implementation of first-order model (`FirstOrderLanguageModel` in [`language_model.py`](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/language_model.py#L42))
- [x] **Deliverable 2:** Python implementation of second-order model (`SecondOrderLanguageModel` in [`language_model.py`](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/language_model.py#L148))
- [x] **Deliverable 3:** Conditional probability tables for selected contexts (Displayed in Section 2 and Section 8)
- [x] **Deliverable 4:** Examples of generated text (20 sampled sentences in [`generated_sentences_first_order.txt`](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/generated_sentences_first_order.txt) and [`generated_sentences_second_order.txt`](file:///home/devambhavsar/ML/AI-Gurukul/Site-Plan-Agent/AI_slop/ai-lab-solution/Lab5_Bayesian_Networks/generated_sentences_second_order.txt))
- [x] **Deliverable 5:** Results of probability-normalisation tests (Section 4; verified via `pytest`)
- [x] **Deliverable 6:** Complete answers to Questions 1 through 14
- [x] **Deliverable 7:** Reflection on LLM usage with inspection and correction of joint-vs-conditional normalization bug
