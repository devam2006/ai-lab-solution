"""
AI Laboratory: Bayesian Networks and Autoregressive Language Models
===================================================================
Implementation of First-Order and Second-Order Autoregressive Language Models
represented as Bayesian Networks with Conditional Probability Tables (CPTs).

Features:
- Pure Python data structures and random sampling (no black-box ML frameworks)
- First-order Markov model: P(X_t | X_{t-1})
- Second-order Markov model: P(X_t | X_{t-2}, X_{t-1})
- Verification of probabilistic invariants (sum of conditional probabilities == 1.0)
- Generation modes: Deterministic (Greedy / Argmax) and Probabilistic (Sampling)
- Quantitative and qualitative comparisons between 1st and 2nd order models
"""

import os
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Special boundary tokens
START_TOKEN = "<START>"
END_TOKEN = "<END>"

# Benchmark training dataset from laboratory specification
RAW_DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenize_sentence(sentence: str) -> List[str]:
    """
    Lowercases and splits sentence into tokens, wrapping with <START> and <END>.
    """
    clean_words = sentence.strip().lower().split()
    return [START_TOKEN] + clean_words + [END_TOKEN]


def prepare_dataset(sentences: List[str] = RAW_DATASET) -> List[List[str]]:
    """
    Tokenizes all sentences in the corpus.
    """
    return [tokenize_sentence(s) for s in sentences]


class FirstOrderLanguageModel:
    """
    First-order Autoregressive Language Model structured as a Bayesian Network:
    X_1 -> X_2 -> X_3 -> ... -> X_T
    estimating P(X_t | X_{t-1}).
    """
    def __init__(self):
        # Counts: transitions[w_prev][w_next] = count
        self.transitions: Dict[str, Counter] = defaultdict(Counter)
        # Probabilities: probabilities[w_prev][w_next] = P(w_next | w_prev)
        self.probabilities: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.vocabulary: set = set()
        self.total_transitions_count: int = 0

    def fit(self, tokenized_sentences: List[List[str]]) -> "FirstOrderLanguageModel":
        """
        Learns transition counts and constructs Conditional Probability Tables (CPTs).
        """
        self.transitions.clear()
        self.probabilities.clear()
        self.vocabulary.clear()
        self.total_transitions_count = 0

        # Step 1: Count transitions between consecutive tokens
        for sentence in tokenized_sentences:
            for w in sentence:
                self.vocabulary.add(w)
            for i in range(len(sentence) - 1):
                w_curr = sentence[i]
                w_next = sentence[i + 1]
                self.transitions[w_curr][w_next] += 1
                self.total_transitions_count += 1

        # Step 2: Compute conditional distribution P(X_t | X_{t-1})
        for w_prev, next_counts in self.transitions.items():
            total = sum(next_counts.values())
            for w_next, count in next_counts.items():
                self.probabilities[w_prev][w_next] = count / total

        return self

    def get_distribution(self, previous_token: str) -> Dict[str, float]:
        """
        Returns the conditional probability distribution P(next_token | previous_token).
        """
        return self.probabilities.get(previous_token, {})

    def predict_argmax(self, previous_token: str) -> Optional[str]:
        """
        Predicts the most probable next token: argmax_w P(w | previous_token).
        """
        dist = self.get_distribution(previous_token)
        if not dist:
            return None
        # In case of tie, picks the first maximum deterministically
        max_prob = max(dist.values())
        candidates = [w for w, p in dist.items() if p == max_prob]
        return candidates[0]

    def sample_next(self, previous_token: str, rng: Optional[random.Random] = None) -> Optional[str]:
        """
        Samples the next token from conditional distribution P(w | previous_token).
        """
        dist = self.get_distribution(previous_token)
        if not dist:
            return None
        words = list(dist.keys())
        probs = list(dist.values())
        if rng:
            return rng.choices(words, weights=probs, k=1)[0]
        return random.choices(words, weights=probs, k=1)[0]

    def generate(self, mode: str = "sample", max_tokens: int = 30, rng: Optional[random.Random] = None) -> List[str]:
        """
        Generates a sequence of tokens starting from <START> until <END>.
        Modes: 'sample' (probabilistic) or 'greedy' (deterministic argmax).
        """
        tokens = [START_TOKEN]
        curr_token = START_TOKEN

        while curr_token != END_TOKEN and len(tokens) < max_tokens:
            if mode == "greedy":
                next_token = self.predict_argmax(curr_token)
            elif mode == "sample":
                next_token = self.sample_next(curr_token, rng=rng)
            else:
                raise ValueError(f"Unknown generation mode: {mode}")

            if next_token is None:
                # Encountered unobserved context; terminate sequence
                break

            tokens.append(next_token)
            curr_token = next_token

        return tokens

    def check_normalization(self, tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Verifies the probabilistic invariant: sum_v P(v | w) == 1.0 for every context w.
        """
        results = {}
        for word, dist in self.probabilities.items():
            total = sum(dist.values())
            results[word] = total
            assert abs(total - 1.0) < tolerance, f"Normalization violation for word '{word}': sum = {total}"
        return results

    def format_cpt_table(self, words: Optional[List[str]] = None) -> str:
        """
        Formats CPT into a clean markdown/ASCII table.
        """
        if words is None:
            words = sorted(self.probabilities.keys())

        headers = ["Current Word (w_i)", "Next Word (w_j)", "Count C(w_i, w_j)", "P(w_j | w_i)"]
        rows = []
        for w in words:
            if w in self.probabilities:
                for next_w, p in sorted(self.probabilities[w].items(), key=lambda x: -x[1]):
                    cnt = self.transitions[w][next_w]
                    rows.append([w, next_w, cnt, f"{p:.4f}"])
        if HAS_TABULATE:
            return tabulate(rows, headers=headers, tablefmt="github")
        return "\n".join([f"{r[0]:<10} -> {r[1]:<10} count: {r[2]:<3} prob: {r[3]}" for r in rows])


class SecondOrderLanguageModel:
    """
    Second-order Autoregressive Language Model:
    X_{t-2} -> X_t <- X_{t-1}
    estimating P(X_t | X_{t-2}, X_{t-1}).
    """
    def __init__(self):
        # Counts: transitions[(w_prev2, w_prev1)][w_next] = count
        self.transitions: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        # Probabilities: probabilities[(w_prev2, w_prev1)][w_next] = P(w_next | w_prev2, w_prev1)
        self.probabilities: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        self.vocabulary: set = set()
        self.first_order_fallback = FirstOrderLanguageModel()
        self.total_triples_count: int = 0

    def fit(self, tokenized_sentences: List[List[str]]) -> "SecondOrderLanguageModel":
        """
        Counts observed triples (w_{t-2}, w_{t-1}, w_t) and builds second-order CPTs.
        """
        self.transitions.clear()
        self.probabilities.clear()
        self.vocabulary.clear()
        self.total_triples_count = 0

        # Also train first order for the very first step P(X_1 | <START>)
        self.first_order_fallback.fit(tokenized_sentences)

        for sentence in tokenized_sentences:
            for w in sentence:
                self.vocabulary.add(w)
            for i in range(len(sentence) - 2):
                ctx = (sentence[i], sentence[i + 1])
                w_next = sentence[i + 2]
                self.transitions[ctx][w_next] += 1
                self.total_triples_count += 1

        # Calculate conditional distribution P(X_t | X_{t-2}, X_{t-1})
        for ctx, next_counts in self.transitions.items():
            total = sum(next_counts.values())
            for w_next, count in next_counts.items():
                self.probabilities[ctx][w_next] = count / total

        return self

    def get_distribution(self, context: Tuple[str, str]) -> Dict[str, float]:
        """
        Returns conditional distribution P(next_token | context).
        """
        return self.probabilities.get(context, {})

    def predict_argmax(self, context: Tuple[str, str]) -> Optional[str]:
        """
        Predicts argmax_w P(w | context).
        """
        dist = self.get_distribution(context)
        if not dist:
            return None
        max_prob = max(dist.values())
        candidates = [w for w, p in dist.items() if p == max_prob]
        return candidates[0]

    def sample_next(self, context: Tuple[str, str], rng: Optional[random.Random] = None) -> Optional[str]:
        """
        Samples next token from P(w | context).
        """
        dist = self.get_distribution(context)
        if not dist:
            return None
        words = list(dist.keys())
        probs = list(dist.values())
        if rng:
            return rng.choices(words, weights=probs, k=1)[0]
        return random.choices(words, weights=probs, k=1)[0]

    def generate(self, mode: str = "sample", max_tokens: int = 30, rng: Optional[random.Random] = None) -> List[str]:
        """
        Generates sequence:
        X_1 ~ P(X_1 | <START>)
        X_2 ~ P(X_2 | <START>, X_1)
        ...
        X_t ~ P(X_t | X_{t-2}, X_{t-1}) until <END>.
        """
        tokens = [START_TOKEN]

        # Step 1: Generate X_1 from P(X_1 | <START>)
        if mode == "greedy":
            first_token = self.first_order_fallback.predict_argmax(START_TOKEN)
        else:
            first_token = self.first_order_fallback.sample_next(START_TOKEN, rng=rng)

        if first_token is None:
            return tokens
        tokens.append(first_token)

        # Subsequent tokens generated conditioned on (X_{t-2}, X_{t-1})
        while tokens[-1] != END_TOKEN and len(tokens) < max_tokens:
            ctx = (tokens[-2], tokens[-1])
            if mode == "greedy":
                next_token = self.predict_argmax(ctx)
            elif mode == "sample":
                next_token = self.sample_next(ctx, rng=rng)
            else:
                raise ValueError(f"Unknown generation mode: {mode}")

            if next_token is None:
                break
            tokens.append(next_token)

        return tokens

    def check_normalization(self, tolerance: float = 1e-6) -> Dict[Tuple[str, str], float]:
        """
        Verifies that sum_v P(v | context) == 1.0 for all observed contexts.
        """
        results = {}
        for ctx, dist in self.probabilities.items():
            total = sum(dist.values())
            results[ctx] = total
            assert abs(total - 1.0) < tolerance, f"Normalization violation for context {ctx}: sum = {total}"
        return results

    def format_cpt_table(self) -> str:
        """
        Formats second-order CPT table.
        """
        headers = ["Context (X_{t-2}, X_{t-1})", "Next Word (X_t)", "Count C(ctx, X_t)", "P(X_t | ctx)"]
        rows = []
        for ctx in sorted(self.probabilities.keys()):
            ctx_str = f"({ctx[0]}, {ctx[1]})"
            for next_w, p in sorted(self.probabilities[ctx].items(), key=lambda x: -x[1]):
                cnt = self.transitions[ctx][next_w]
                rows.append([ctx_str, next_w, cnt, f"{p:.4f}"])
        if HAS_TABULATE:
            return tabulate(rows, headers=headers, tablefmt="github")
        return "\n".join([f"{r[0]:<20} -> {r[1]:<10} count: {r[2]:<3} prob: {r[3]}" for r in rows])


def compare_models(m1: FirstOrderLanguageModel, m2: SecondOrderLanguageModel, num_samples: int = 100):
    """
    Compares First-Order and Second-Order models quantitatively and qualitatively:
    - Number of distinct parameters
    - Number of zero-probability contexts
    - Diversity of generated sentences
    - Qualitative coherence
    """
    vocab_size = len(m1.vocabulary)

    # First-order parameters
    m1_distinct_params = sum(len(dist) for dist in m1.probabilities.values())
    m1_total_contexts = vocab_size
    m1_observed_contexts = len(m1.probabilities)
    m1_zero_contexts = m1_total_contexts - m1_observed_contexts

    # Second-order parameters
    m2_distinct_params = sum(len(dist) for dist in m2.probabilities.values())
    m2_total_contexts = vocab_size * vocab_size
    m2_observed_contexts = len(m2.probabilities)
    m2_zero_contexts = m2_total_contexts - m2_observed_contexts

    # Diversity test: generate samples and measure uniqueness
    rng = random.Random(42)
    m1_samples = [" ".join(m1.generate(mode="sample", rng=rng)[1:-1]) for _ in range(num_samples)]
    m2_samples = [" ".join(m2.generate(mode="sample", rng=rng)[1:-1]) for _ in range(num_samples)]

    m1_unique = len(set(m1_samples))
    m2_unique = len(set(m2_samples))

    # Coherence check: check for semantic absurdities (e.g., "sat on the cat/dog" or "ran to the mat/rug")
    invalid_patterns = ["on the cat", "on the dog", "on the park", "to the mat", "to the rug", "the mat sat", "the rug sat"]
    m1_incoherent_count = sum(1 for s in m1_samples if any(pat in s for pat in invalid_patterns))
    m2_incoherent_count = sum(1 for s in m2_samples if any(pat in s for pat in invalid_patterns))

    comparison_data = {
        "m1_params": m1_distinct_params,
        "m2_params": m2_distinct_params,
        "m1_observed_contexts": m1_observed_contexts,
        "m2_observed_contexts": m2_observed_contexts,
        "m1_zero_contexts": m1_zero_contexts,
        "m2_zero_contexts": m2_zero_contexts,
        "m1_total_contexts": m1_total_contexts,
        "m2_total_contexts": m2_total_contexts,
        "m1_unique_pct": (m1_unique / num_samples) * 100,
        "m2_unique_pct": (m2_unique / num_samples) * 100,
        "m1_incoherent_pct": (m1_incoherent_count / num_samples) * 100,
        "m2_incoherent_pct": (m2_incoherent_count / num_samples) * 100,
    }
    return comparison_data


def run_laboratory_suite():
    """
    Executes the complete experimental suite required by BN_lab.pdf and prints outputs.
    """
    print("=" * 70)
    print("AI LABORATORY: BAYESIAN NETWORKS & AUTOREGRESSIVE LANGUAGE MODELS")
    print("=" * 70)

    # 1. Dataset Preparation
    dataset = prepare_dataset()
    print(f"\n[+] Loaded Dataset: {len(dataset)} sentences")
    for i, s in enumerate(dataset, 1):
        print(f"  Sentence {i}: {' '.join(s)}")

    # 2. Fit First-Order Model
    m1 = FirstOrderLanguageModel()
    m1.fit(dataset)
    print(f"\n[+] First-Order Model Trained:")
    print(f"  Vocabulary size: {len(m1.vocabulary)} tokens")
    print(f"  Total observed transitions: {m1.total_transitions_count}")
    print(f"  Number of distinct contexts: {len(m1.probabilities)}")

    # 3. Question 3: Conditional Distributions for Key Words
    target_words = ["the", "cat", "dog", "sat", "ran"]
    print("\n[+] Conditional Probability Tables (CPTs) for Selected Words (Question 3):")
    print(m1.format_cpt_table(target_words))

    # 4. Question 8: Test Probability Invariants
    print("\n[+] Testing Probability Invariant sum_v P(v | w) == 1.0 (Part VII / Question 8):")
    norm_results = m1.check_normalization()
    for word, total in sorted(norm_results.items()):
        status = "PASSED" if abs(total - 1.0) < 1e-6 else "FAILED"
        print(f"  Context w = '{word:<8}': Sum = {total:.6f} [{status}]")

    # 5. Question 9: Predicting Next Word (Argmax) for 5 contexts
    test_words = ["the", "cat", "dog", "sat", "ran", "on", "to"]
    print("\n[+] Next-Word Distribution & Argmax Predictions (Part VIII / Question 9):")
    for w in test_words:
        dist = m1.get_distribution(w)
        argmax_word = m1.predict_argmax(w)
        dist_str = ", ".join([f"{k}: {v:.3f}" for k, v in dist.items()])
        print(f"  P(X_{{t+1}} | X_t = '{w}'): [{dist_str}] -> argmax: '{argmax_word}'")

    # 6. Part IX: Generate 20 Sentences using Sampling
    print("\n[+] Generating 20 Sentences using Probabilistic Sampling (Part IX):")
    rng = random.Random(1337)
    sampled_20 = []
    for i in range(1, 21):
        sent_tokens = m1.generate(mode="sample", rng=rng)
        sent_str = " ".join(sent_tokens)
        sampled_20.append(sent_str)
        print(f"  {i:2d}. {sent_str}")

    # Save to file
    out_dir = os.path.dirname(os.path.abspath(__file__))
    f1_path = os.path.join(out_dir, "generated_sentences_first_order.txt")
    with open(f1_path, "w") as f:
        for s in sampled_20:
            f.write(s + "\n")
    print(f"  -> Saved 20 sentences to '{f1_path}'")

    # 7. Part X: Deterministic vs Probabilistic Generation (Question 10)
    print("\n[+] Mode A (Greedy / Deterministic) Generation (5 sentences):")
    for i in range(1, 6):
        print(f"  {i}. {' '.join(m1.generate(mode='greedy'))}")

    print("\n[+] Mode B (Sampling / Probabilistic) Generation (5 sentences):")
    for i in range(1, 6):
        print(f"  {i}. {' '.join(m1.generate(mode='sample', rng=rng))}")

    # 8. Part XI & XII: Fit Second-Order Model
    m2 = SecondOrderLanguageModel()
    m2.fit(dataset)
    print("\n[+] Second-Order Model Trained:")
    print(f"  Total observed triples: {m2.total_triples_count}")
    print(f"  Number of observed contexts (X_{{t-2}}, X_{{t-1}}): {len(m2.probabilities)}")

    print("\n[+] Second-Order CPT Table (Sample contexts):")
    print(m2.format_cpt_table())

    # Second-order normalization test
    m2_norm = m2.check_normalization()
    print(f"\n[+] Second-Order Invariant Check: All {len(m2_norm)} contexts sum to exactly 1.0 [PASSED]")

    # 9. Second-Order Generation Examples
    print("\n[+] Generating Sentences with Second-Order Model (Sampling):")
    sampled_m2_20 = []
    for i in range(1, 21):
        sent_tokens = m2.generate(mode="sample", rng=rng)
        sent_str = " ".join(sent_tokens)
        sampled_m2_20.append(sent_str)
        print(f"  {i:2d}. {sent_str}")

    f2_path = os.path.join(out_dir, "generated_sentences_second_order.txt")
    with open(f2_path, "w") as f:
        for s in sampled_m2_20:
            f.write(s + "\n")
    print(f"  -> Saved 20 sentences to '{f2_path}'")

    # 10. Part XIII: Model Comparison
    comp = compare_models(m1, m2, num_samples=100)
    print("\n" + "=" * 70)
    print("PART XIII: MODEL COMPARISON SUMMARY (1st-Order vs 2nd-Order)")
    print("=" * 70)
    comp_table = [
        ["Metric", "First-Order Model", "Second-Order Model"],
        ["Theoretical Context Space (|V|^k)", f"{comp['m1_total_contexts']}", f"{comp['m2_total_contexts']}"],
        ["Observed Contexts", f"{comp['m1_observed_contexts']}", f"{comp['m2_observed_contexts']}"],
        ["Zero-Probability Contexts", f"{comp['m1_zero_contexts']} ({(comp['m1_zero_contexts']/comp['m1_total_contexts'])*100:.1f}%)", f"{comp['m2_zero_contexts']} ({(comp['m2_zero_contexts']/comp['m2_total_contexts'])*100:.1f}%)"],
        ["Distinct Non-Zero Parameters", f"{comp['m1_params']}", f"{comp['m2_params']}"],
        ["Sample Diversity (Unique % / 100)", f"{comp['m1_unique_pct']:.1f}%", f"{comp['m2_unique_pct']:.1f}%"],
        ["Semantic Incoherence Rate", f"{comp['m1_incoherent_pct']:.1f}%", f"{comp['m2_incoherent_pct']:.1f}%"],
    ]
    if HAS_TABULATE:
        print(tabulate(comp_table, headers="firstrow", tablefmt="github"))
    else:
        for row in comp_table:
            print(f"{row[0]:<35} | {row[1]:<20} | {row[2]:<20}")


if __name__ == "__main__":
    run_laboratory_suite()
