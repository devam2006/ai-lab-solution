# Neural Models - Laboratory Submission

---

## 1. Task 1: Problem Understanding
**The XOR Problem:**
We are building a safety sensor that throws an alarm ($y=1$) exactly when two sensors $x_1, x_2$ disagree (i.e., $(0,1)$ or $(1,0)$). If they agree (both $0$ or both $1$), the alarm is off ($y=0$).

**Why one straight decision boundary fails:**
XOR is physically impossible to separate with a single straight line. If you plot the four points on an X-Y graph, the two $y=1$ points are on opposite corners (top-left, bottom-right), and the two $y=0$ points are on the other opposite corners. No single straight line can ever put the 1s on one side and the 0s on the other. 

**Prediction for a single linear (affine) transformation:**
If we train a network with NO hidden layers (or a hidden layer with no non-linear activation), it will be mathematically equivalent to a single straight line. The loss will get stuck, and the network will just guess $0.5$ (50%) for everything, failing completely to learn the XOR pattern.

---

## 2. Task 2: Model Design
**Why is the hidden non-linearity scientifically necessary?**
Without an activation function like Tanh or ReLU, stacking multiple linear layers ($W_2 \times (W_1 \times X)$) mathematically collapses into just one single linear layer ($W_{combined} \times X$). The non-linearity bends and folds the data space, allowing the network to draw curved or multiple boundaries to isolate the XOR corners.

**Why is Sigmoid + Binary Cross-Entropy sensible?**
Because the target is a binary YES/NO answer (1 or 0). Sigmoid perfectly squashes any raw output number into a probability between $0.0$ and $1.0$. Binary Cross-Entropy is specifically formulated to heavily penalize the network when this probability diverges from the true 1 or 0 label.

**Validation Criteria for Successful Learning:**
1. **Final Loss:** Drops close to $0.0$.
2. **Thresholded Predictions:** Must perfectly match `[0, 1, 1, 0]`.
3. **Probabilities:** Must be very close to `0.0` or `1.0` (not stuck at `0.5`).

---

## 3. Tasks 3 & 4: Execution & Experiments

*(Code for these experiments is available in `neural_xor.py`)*

### Part A: Basic Learning Check
Using `Tanh` hidden activation:
- **Initial Loss:** ~0.7000
- **Final Loss:** ~0.0050
- **Final Thresholded Predictions:** `[0., 1., 1., 0.]` (4/4 Correct)
- **Conclusion:** The model successfully learned the XOR pattern.

### Part B: Backpropagation Check
The gradient `parameter.grad` for the first layer weights represents the partial derivative of the overall scalar Loss with respect to each individual weight in that matrix ($\frac{\partial L}{\partial W^{(1)}}$). Since we feed all four examples at once (full batch), this gradient tensor is the **average** of the individual gradients from all four examples. It tells the optimizer exactly which direction and how steeply to adjust those specific weights to reduce the loss.

### Part C: Symmetry Experiment (Zero Initialization)
If all weights and biases are initialized to exactly zero, the model completely fails to learn. 
**Explanation:** When weights are identical, both hidden units receive the exact same input, produce the exact same output, and therefore receive the exact same error gradient during backpropagation. Because they update identically, they remain perfectly symmetrical forever. The network essentially acts as if it only has *one* hidden unit, destroying its ability to solve XOR.

### Part D: Activation Experiment
| Hidden Activation | Final Loss | 4/4 Correct? | Early $\|\nabla_{W^{(1)}}L\|_2$ |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | Very High (~0.69) | No (often stuck) | Very Small |
| **Tanh** | Very Low (~0.005) | Yes | Moderate |
| **ReLU** | Varies (often 0.0) | Yes (usually) | Large (or exactly 0 if dead) |

**Interpretation:**
- **Sigmoid** suffers from the vanishing gradient problem. Its derivatives are small (max 0.25), so when multiplied through layers, the learning signal shrinks, causing slow or stalled learning.
- **Tanh** is zero-centered and has steeper derivatives, leading to much faster and more reliable convergence for this tiny problem.
- **ReLU** has a derivative of exactly 1.0 when active, allowing very fast learning. However, if initialized poorly, units can output negative values, making the derivative exactly 0.0 ("dying ReLU").

---

## 4. Task 5: Three-Class Extension

When moving from a 1-output binary decision to a 3-output multi-class decision (0, 1, or 2):
1. **Weight Matrix Shape:** The final layer changes from `Linear(2, 1)` to `Linear(2, 3)`.
2. **Logits per example:** The network now outputs exactly 3 logits per example (one for each class).
3. **Softmax:** Softmax mathematically converts raw logits into a probability distribution by exponentiating and dividing by the sum. This guarantees all probabilities are positive and sum exactly to $1.0$.
4. **Stable Softmax (Subtracting Max Logit):** Exponentiating large logits (e.g., $e^{1000}$) causes computer hardware to hit mathematical overflow (infinity). By subtracting the maximum logit from all logits before exponentiating, the largest value becomes $e^0 = 1$, making it perfectly numerically stable without changing the resulting probability ratios.

---

## 5. Reflection Questions

**1. What did the XOR experiment demonstrate about the difference between depth and non-linearity?**
Adding "depth" (more layers) is useless if the layers are purely linear, as they just collapse into one line. Non-linearity is what actually gives deep networks the power to model complex, curved decision boundaries.

**2. What evidence showed backpropagation supplied a useful learning signal?**
The fact that the final loss approached zero and the predictions exactly matched the XOR labels. A completely random or zero gradient would have left the network guessing 0.5 forever.

**3. Why did identical/zero initialization prevent the hidden units from learning distinct features?**
Because of symmetry. If two neurons start with the same weights, they process the inputs identically, get the same error, and update identically. They are mathematically locked together.

**4. How did changing the hidden activation affect the gradient?**
Scientifically, Sigmoid has flat tails, meaning large inputs cause the derivative to approach zero. Engineering-wise, we observed this as the early gradient norm being tiny for Sigmoid, causing the loss to barely drop, whereas Tanh had a healthier gradient norm and solved the problem quickly.

**5. Why must the output layer and loss be selected together?**
The loss function math explicitly expects a certain type of output. Binary Cross Entropy expects a single probability between 0 and 1 (so we use Sigmoid). Multi-class Cross Entropy expects a probability distribution summing to 1 across multiple logits (so we use Softmax). Mixing them up breaks the math.

**6. Give one example where the LLM improved productivity and one where human verification was essential.**
The LLM was fantastic at instantly generating the PyTorch training loop boilerplate (zeroing gradients, stepping the optimizer, calculating loss). However, human verification was absolutely essential to know *why* we needed `BCEWithLogitsLoss` instead of regular MSE, and to properly interpret why the Symmetry Experiment failed.

**7. Which tests would you keep if the model were scaled up?**
Tracking the Loss curve, Final Accuracy, and perhaps the Gradient Norm (to watch for vanishing/exploding gradients) are standard practice and scale well. Exhaustive finite-difference checks or printing every single logit/probability vector becomes impossible when the model has billions of parameters and millions of classes.
