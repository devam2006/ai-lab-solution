# Reinforcement Learning and Deep RL — Task Answers

**Course:** Artificial Intelligence (CS F407)  
**Lab:** [RL and Deep RL handout](RL_lab_2026.pdf)  
**Supporting work:** [report and complete joint trajectory](submission.md), [program](warehouse.py), [tests](test_warehouse.py), [recorded results](results.json)

## Task 1: Formulate the MDP

| Component | Specification |
|---|---|
| State | Robot position `(r,c)` on a free cell of a 5×5 grid. Obstacles are `(0,2)`, `(1,2)`, `(3,2)`, `(4,2)`. |
| Actions | Up, down, left, right, wait, encoded 0–4 respectively. |
| Transition | Deterministic. A move changes the position by one cell; a boundary or obstacle attempt leaves it unchanged. Wait also leaves it unchanged. |
| Reward | +10 on reaching the goal; otherwise −1 for an invalid move; otherwise −0.1, including valid wait. |
| Initial state | `(2,0)` |
| Terminal outcome | Reaching `(2,4)`. An unfinished rollout is truncated at 40 steps, not terminated. Discount factor γ = 0.95. |

**(a)** The current row and column specify the state. **(b)** They suffice because the map, goal and rules are fixed: they determine the next position and reward for each action. **(c)** The transitions are deterministic. **(d)** A shortest route is `(2,0) → (2,1) → (2,2) → (2,3) → (2,4)`, taking four steps. **(e)** Search uses a model to compute a route; Q-learning estimates action values from trial-and-error experience. The learner is not given the route.

## Task 2: Plan the learning agent

- Represent the grid by its size and obstacle set, and the robot by a `(row, column)` tuple. `reset` sets the position and step counter. `step` applies an action and returns next state, reward, termination, truncation and diagnostics.
- Use a 25×5 Q-table, with state index `5*r+c`. Obstacle entries are unused. Start with zero values and α = 0.2.
- In training, select a uniformly random action with probability ε; otherwise select a maximum-valued action, breaking ties randomly. Decrease ε linearly from 1 to 0.05 over 1,200 episodes.
- Use `r` as the TD target for true termination; otherwise use `r + γ max_a Q(s',a)`. A truncated transition still uses its final next state for bootstrapping. Stop a rollout on either termination or truncation.
- Record each episode's return, success, ε, invalid moves and timeout. Evaluate separately with ε = 0, no updates and a fixed first-maximum tie break. Record the trajectory and length.

## Task 3: LLM-assisted Q-learning

The user's request was to complete the supplied lab and push it to the repository. The implementation was guided by the handout's exact environment, rewards, separate termination and truncation, tests, and prescribed learning settings. The assistant produced [warehouse.py](warehouse.py), with separate environment, action selection, update, training and evaluation functions. Inspect a single transition without training using `python warehouse.py --inspect ROW COL ACTION`.

The report's implementation section summarizes the instructions given to the assistant. The design was derived from the handout during this interaction; it would be inaccurate to claim the student independently prepared it before prompting. The checks below show what the program actually does.

## Task 4: Test the environment and the Q update

All four automated test groups passed. The single-robot tests set the position directly before each action:

| Starting position | Action | Observed next state | Reward | Terminated? |
|---|---|---|---:|---|
| `(0,0)` | Up | `(0,0)` | −1 | No |
| `(1,1)` | Right | `(1,1)` | −1 | No |
| `(2,1)` | Right | `(2,2)` | −0.1 | No |
| `(2,1)` | Wait | `(2,1)` | −0.1 | No |
| `(2,3)` | Right | `(2,4)` | +10 | Yes |

Forty waits from `(0,0)` produced truncation without termination. For the synthetic update `Q(s,a)=0.4`, `r=1`, `max Q(s',·)=2`, α=0.2, γ=0.95:

`target = 1 + 0.95 × 2 = 2.9`; `new Q = 0.4 + 0.2 × (2.9 − 0.4) = 0.9`.

For a true terminal reward of 5, `target = 5` and `new Q = 0.4 + 0.2 × (5 − 0.4) = 1.32`. With truncation rather than termination, the target remains 2.9 and the new entry 0.9. **Only true termination ignores the next-state value.**

| Concept | Location in [warehouse.py](warehouse.py) |
|---|---|
| State, actions, and movement | `MOVES`, `index`, `move`, `Warehouse` |
| Reward and termination | `Warehouse.step` |
| Epsilon-greedy choice | `choose` |
| TD target and table update | `q_update` |
| Training and greedy evaluation | `train_table`, `evaluate` |

## Task 5: Train and investigate exploration

Q-learning used 1,500 episodes and seed 7. The [return and success plots](q_training.png) show 50-episode moving averages. Greedy evaluation produced the four-step route `(2,0) → (2,1) → (2,2) → (2,3) → (2,4)`, matching the shortest route, with zero invalid moves. The [trajectory plot](single_trajectories.png) shows the grid.

| Measure | Uniform random policy, 100 runs | Greedy Q-policy, one run |
|---|---|---|
| Goal reached | 15/100 (15%) | Yes |
| Steps to goal | Mean 25.4 among 15 successes | 4 |
| Timeouts | 85/100 (85%) | No |
| Invalid moves | 970 total; 9.7 per run | 0 |

The random runs use seeds 0–99 and the same 40-step cutoff. **(a)** Training remains noisy because exploratory actions continue even after a good route has been learned. **(b)** A high exploratory training return does not prove that the separately evaluated greedy policy is good. **(c)** α sets the size of each value update; ε sets the probability of choosing a random action. The optional no-exploration experiment was not run.

## Task 6: Deep Q-Network

The map, transitions, rewards, start, goal, discount and stopping rules remain unchanged. A neural network replaces the table: a 25-element one-hot state enters `25 → 64 ReLU → 64 ReLU → 5 linear Q-values`. The outputs are values, not probabilities. The code uses a replay buffer of 20,000 transitions, random batches of 64, 500 initial collection steps, an optimizer update every four environment steps, and a target copy every 250 optimizer updates. Adam uses learning rate 0.001 and mean squared TD error. Targets are detached; termination masks the continuation term, truncation does not.

The test batch gives targets **2.9, 5, 2.9** for non-terminal, terminal and truncated transitions. The test checks that `gather` selects the recorded action's online Q-value and that gradients do not flow into the target. In `train_dqn`, `replay` is the buffer, `rng.sample` draws a batch, `targets` computes the target, `optimizer.step()` updates the online network, and `target.load_state_dict` synchronizes it.

With seed 7 and 12,000 steps, greedy DQN reached the goal in four steps with zero invalid moves, following the same route as the table. The different training budgets do not establish which method is more sample-efficient.

## Task 7: Two robots and collision avoidance

**Design.** A's position alone cannot reveal where B is or whether their proposed moves collide. The state is ordered `(pA,pB)`, encoded as two concatenated 25-element one-hot vectors. There are 25 joint actions, encoded `5*aA+aB` and decoded by `divmod(j,5)`; the network has 50 inputs and 25 linear outputs.

Proposals are computed simultaneously from the old positions. Invalid moves become stays; a robot at its goal stays and occupies the cell. Equal proposed cells or a swap cause collision, −10 reward, retained old positions and termination. Entering a waiting robot's cell collides, but following into a vacated cell is allowed if there is no swap. Otherwise, the reward is `−0.1 − b + 5k + 5I`, where `b` is the count of invalid moves by robots not already at goal, `k` is newly arrived robots, and `I` indicates both are now at their goals. Both goals terminate; unfinished runs truncate at 40 steps.

One possible **designed example**, not a learned trajectory, is shown below. A yields at `(1,1)` so B can pass. Positions are after simultaneous moves.

| Step | A | B |
|---:|---|---|
| 0 | `(2,0)` | `(2,4)` |
| 1 | `(2,1)` | `(2,3)` |
| 2 | `(1,1)` | `(2,2)` |
| 3 | `(1,1)` | `(2,1)` |
| 4 | `(2,1)` | `(2,0)` |
| 5 | `(2,2)` | `(2,0)` |
| 6 | `(2,3)` | `(2,0)` |
| 7 | `(2,4)` | `(2,0)` |

**Tests:**

| Old A, B | Actions A, B | Observed result |
|---|---|---|
| `(2,1)`, `(2,3)` | Right, left | Same-cell collision; old positions retained; −10; terminate |
| `(2,1)`, `(2,2)` | Right, left | Swap collision; old positions retained; −10; terminate |
| `(2,1)`, `(2,2)` | Right, right | `(2,2)`, `(2,3)`; −0.1; no termination |
| `(2,1)`, `(2,2)` | Up, wait | `(1,1)`, `(2,2)`; −0.1; no termination |

An already-arrived robot stayed at its goal on repeated actions, with no repeated arrival reward. Entering a waiting robot's cell also produced a collision. The test suite checked the combined two-arrival reward of 14.9.

**Learned policy:** With 60,000 steps and seed 7, A entered the passage first, reaching `(2,2)` on step 2. B moved aside through `(3,4)` and `(4,4)` to `(4,3)`, where it stayed. A reached its goal, but B did not: the run timed out after 40 steps with no collision and zero invalid moves. See the [time-labelled plot](two_trajectory.png) and the [position table for every step](submission.md#5-experiments-and-discussion). Avoiding collision alone is insufficient; both deliveries must finish. Three robots would have `5³ = 125` joint actions.

## Task 8: Evaluation and reflection

| Controller | Budget / seed | Default-start outcome | Steps if successful | Invalid moves |
|---|---|---|---:|---:|
| Q-table | 1,500 episodes / 7 | Success | 4 | 0 |
| Single DQN | 12,000 steps / 7 | Success | 4 | 0 |
| Joint DQN | 60,000 steps / 7 | Timeout; no collision | — | 0 |

Without retraining, both single-robot policies succeeded from `(0,0)` in six steps with zero invalid moves. From A=`(0,0)`, B=`(4,4)`, the joint policy timed out without collision and made one invalid move. These are individual deterministic rollouts, not independent trials. The two tasks' raw returns should not be compared directly because their reward definitions differ.

1. **Design specified:** The handout specified the state, actions, map, rewards, stopping rules, network architecture and training settings; the assistant translated these into the program. The student did not provide a separate prior design in this interaction.
2. **Changes to rules:** The implementation was checked against the specified rewards, termination and truncation rules and the requested algorithm details. No intentional change is identified.
3. **Errors found:** The recorded test run passed all four groups. It did not establish that a prior code error was found; the failed joint rollout is a learning outcome, not evidence that the tested collision rules were incorrect.
4. **Replay and target network:** Yes. `train_dqn` uses a replay buffer, samples batches, and periodically copies the online weights into a separate target network.
5. **Functions to understand:** `move` and `Warehouse.step` implement the environment; `q_update` implements tabular TD learning; `train_dqn` handles replay, targets and optimizer steps; `evaluate` runs a greedy policy. The student should review these before claiming personal understanding.
6. **Changes and evidence:** The implemented separation between goal termination and the 40-step truncation is checked by the environment, tabular update and DQN target tests. There is no documented earlier correction to claim as an observed fix. The tests cover the listed cases rather than every possible state.

### Optional: RLHF connection

People could compare recorded trajectory pairs and choose the one they prefer, for example because it avoids close encounters or unnecessary waiting. A reward model trained on these comparisons would predict a score or preference for trajectories; an RL agent could optimize against that learned signal. Merely editing the hand-written collision penalty does not train a reward model: it changes a rule directly, without learning a reward from preference data. No RLHF training was performed here.

**Reproducibility:** Python 3.14.3, NumPy 2.5.4, PyTorch 2.15.0+cpu, Matplotlib 3.11.2; training seed 7. Run `python -m unittest -v test_warehouse` and `python warehouse.py` from this directory. Results may vary across software versions or hardware.

## Further explanation: reading the learning results

For a single nonterminal transition with reward `−0.1` and best next value 4, the Q-learning target is `−0.1 + 0.95×4 = 3.7`. A truncation after that transition still uses 3.7: the time limit ends data collection, not the underlying task. A true terminal transition instead uses its reward alone. DQN applies the same distinction to a sampled batch; `gather` compares the target with the value for the action actually taken, not the maximum online value over all actions.

In the two-robot task, the collision penalty takes priority over all other reward terms. Without a collision, if exactly one robot newly reaches its goal and neither makes an invalid move, the team reward is `−0.1 + 5 = 4.9`; if both newly arrive on the same step it is `−0.1 + 10 + 5 = 14.9`. A robot already at its goal cannot collect the arrival bonus again. The observed joint timeout illustrates the difference between avoiding a collision and completing the team objective; a single failed seed does not prove that DQN cannot learn the task.
