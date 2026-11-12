# Reinforcement Learning and Deep RL - Laboratory Submission

**Course:** Artificial Intelligence (CS F407)  
**Files:** [implementation](warehouse.py), [tests](test_warehouse.py), [experiment results](results.json), [lab handout](RL_lab_2026.pdf)

## 1. MDP and design

The state is the robot's cell `(r,c)` in a 5×5 grid, excluding the four obstacles `(0,2), (1,2), (3,2), (4,2)`. The fixed initial state is `(2,0)` and the goal is `(2,4)`. Actions 0–4 are up, down, left, right and wait. Transitions are deterministic: an out-of-bounds or obstacle move leaves the position unchanged; wait is valid. Reward is +10 on reaching the goal, otherwise −1 for an invalid move, otherwise −0.1. The discount is 0.95. The goal terminates; a 40-step collection limit truncates, and still bootstraps from the last observed next state.

Position suffices because the map, goal and rules are fixed and the next state and reward depend only on the present position and action. The shortest route is `(2,0) → (2,1) → (2,2) → (2,3) → (2,4)`, four actions. Search computes a route from a model; Q-learning estimates action values from experienced transitions, without a supplied route or demonstrations.

`Warehouse.reset` sets the position and counter; `step` returns the next state, reward, termination, truncation and diagnostics. The table is 25×5, indexed by `5*r+c`; obstacle rows are unused. Training starts at zero, uses α=0.2 and epsilon-greedy action choice with uniform random tie-breaking, and stops on termination or truncation. The next-state maximum is omitted only at true termination. Training records return, success, epsilon, invalid moves and timeouts. Evaluation disables learning and exploration and uses the first maximum as a fixed tie-break. `--inspect ROW COL ACTION` examines one step without training.

## 2. Prompts and implementation

The instruction to the assistant was: “Implement the supplied RL lab according to its exact 5×5 map, actions, rewards, cutoff and tests. Use NumPy Q-learning and PyTorch DQN with replay and a target network, train and evaluate the single and simultaneous two-robot tasks, and record actual results. Do not use a planner or prevent collisions by filtering actions.” The handout specifies the architecture and hyperparameters implemented below. No claim is made that a human wrote a design before this interaction. The assistant implemented the design and tests; the checks below document what was verified. No earlier correction or error is claimed without evidence.

For DQN the environment is unchanged; a 25-element one-hot state enters a 25→64→64→5 network with ReLU hidden layers and linear Q outputs, not probabilities. A 20,000-transition replay buffer supplies random batches of 64. After 500 collected steps, Adam (0.001) updates every four steps using mean squared TD error; the target is copied every 250 optimizer updates. Targets are detached and mask only true termination. The recorded action is selected with `gather` before computing the loss. Epsilon decreases linearly from 1 to 0.05 over 80% of training. The table uses 1,500 episodes and DQN uses 12,000 environment steps; both use seed 7. The joint DQN uses 60,000 steps, seed 7, the same DQN settings, 50 inputs and 25 outputs. Budgets differ, so this is not a sample-efficiency comparison.

## 3. Pre-training checks

Run `python -m unittest -v test_warehouse` from this directory. The tests set each state directly: boundary and obstacle moves stay with −1; the doorway move succeeds with −0.1; waiting stays with −0.1; entering the goal returns +10 and terminates; 40 waits truncate. For a synthetic update with Q=0.4, r=1 and max next Q=2, the target is 2.9 and the updated value is 0.9. With terminal r=5, the target is 5 and the updated value is 1.32. Truncation uses the first result, not the terminal target. The DQN test checks targets `[2.9, 5, 2.9]`, action gathering and the absence of gradients through the target.

For simultaneous actions the tests cover same-cell and swap collisions, following into a vacated cell, a move beside a waiting robot, an attempted move into a waiting robot, and repeated waiting by an already-arrived robot. They also check the two-arrival reward. See the test code for exact inputs and assertions.

## 4. Two robots

The ordered pair of positions is necessary: A's position alone cannot identify B's location, its proposed move or a potential collision. Two 25-element one-hot vectors are concatenated; joint action `5*a_A+a_B` has 25 possibilities and is decoded with `divmod`. Both proposals use the same old state. Invalid moves stay and count as penalties, while robots at their goal remain there. Equal proposed cells and swaps cause a collision, return −10, retain old positions and terminate. Following into a vacated cell is allowed. Otherwise reward is `−0.1 − invalid_count + 5*new_arrivals + 5*both_arrived`; reaching both goals terminates. Arrival is rewarded once. No action filter or planner is used by the controller.

An illustrative feasible route, separate from the learned policy, has A yield at `(1,1)`: first both move toward the door, A moves up while B moves into the door; B then moves left while A waits, B moves left to its goal while A moves down; A then crosses to its goal while B stays. Both use simultaneous steps and never exchange cells. Avoiding collision alone is not success: both deliveries must finish. Three robots would require `5³ = 125` joint actions.

## 5. Experiments and discussion

All four test groups passed. Experiments used Python 3.14.3, NumPy 2.5.4, PyTorch 2.15.0+cpu and Matplotlib 3.11.2 on CPU. Run `python warehouse.py` from this directory to reproduce, followed by `python finish_report.py` to rebuild this result section. Raw episode logs and [results.json](results.json) accompany the report. Seeds 0–99 were used for the 100 random-policy episodes.

| Default start | Budget / seed | Result | Steps to goal | Invalid moves |
|---|---|---|---:|---:|
| Q-table | 1,500 episodes / 7 | Success | 4 | 0 |
| Single DQN | 12,000 steps / 7 | Success | 4 | 0 |
| Joint DQN | 60,000 steps / 7 | Timeout, no collision | — | 0 |

The random policy succeeded in 15/100 runs, timed out in 85/100, averaged 25.4 steps among successes, and made 970 invalid moves in total (9.7 per run). Both single-robot greedy policies followed the four-step shortest route, shown in [the trajectory plot](single_trajectories.png). [Training curves](q_training.png) use a 50-episode moving average; [training data](q_training.csv) includes epsilon, invalid moves and timeouts.

From `(0,0)`, both single policies succeeded in six steps with zero invalid moves. From A=`(0,0)`, B=`(4,4)`, the joint controller timed out at 40 steps without collision, with one invalid move. These are single deterministic rollouts without retraining, not independent trials.

The default joint run failed: A entered the passage first, reaching `(2,2)` at step 2. B moved aside through `(3,4)` and `(4,4)` to `(4,3)`, where it stayed while A reached its goal. B never completed its delivery. The [trajectory plot](two_trajectory.png) labels time steps; repeated positions are shown as ranges. Positions after every simultaneous step are below:

| Step | A | B |
|---:|---|---|
| 0 | (2, 0) | (2, 4) |
| 1 | (2, 1) | (3, 4) |
| 2 | (2, 2) | (4, 4) |
| 3 | (2, 3) | (4, 3) |
| 4 | (2, 4) | (4, 3) |
| 5 | (2, 4) | (4, 3) |
| 6 | (2, 4) | (4, 3) |
| 7 | (2, 4) | (4, 3) |
| 8 | (2, 4) | (4, 3) |
| 9 | (2, 4) | (4, 3) |
| 10 | (2, 4) | (4, 3) |
| 11 | (2, 4) | (4, 3) |
| 12 | (2, 4) | (4, 3) |
| 13 | (2, 4) | (4, 3) |
| 14 | (2, 4) | (4, 3) |
| 15 | (2, 4) | (4, 3) |
| 16 | (2, 4) | (4, 3) |
| 17 | (2, 4) | (4, 3) |
| 18 | (2, 4) | (4, 3) |
| 19 | (2, 4) | (4, 3) |
| 20 | (2, 4) | (4, 3) |
| 21 | (2, 4) | (4, 3) |
| 22 | (2, 4) | (4, 3) |
| 23 | (2, 4) | (4, 3) |
| 24 | (2, 4) | (4, 3) |
| 25 | (2, 4) | (4, 3) |
| 26 | (2, 4) | (4, 3) |
| 27 | (2, 4) | (4, 3) |
| 28 | (2, 4) | (4, 3) |
| 29 | (2, 4) | (4, 3) |
| 30 | (2, 4) | (4, 3) |
| 31 | (2, 4) | (4, 3) |
| 32 | (2, 4) | (4, 3) |
| 33 | (2, 4) | (4, 3) |
| 34 | (2, 4) | (4, 3) |
| 35 | (2, 4) | (4, 3) |
| 36 | (2, 4) | (4, 3) |
| 37 | (2, 4) | (4, 3) |
| 38 | (2, 4) | (4, 3) |
| 39 | (2, 4) | (4, 3) |
| 40 | (2, 4) | (4, 3) |

Training return can remain noisy because epsilon continues to produce exploratory moves even after a good route is learned. A high training return does not guarantee that the greedy policy is good, since the return describes exploratory episodes and depends on the visited states. Alpha controls the size of a value update; epsilon controls how often a random action is selected. A policy trained from a fixed start may not generalize to unvisited states. Repeating a deterministic rollout is not an independent trial. Single- and two-robot raw returns have different reward definitions and should not be compared directly.

The environment (`move`, `Warehouse.step`), table update (`q_update`), selection (`choose`), replay and target updates (`train_dqn`), and evaluation (`evaluate`) are distinct and can be inspected independently. The code uses the requested replay and target network. Tests provide evidence for the enumerated cases, not proof that every possible state or learned behavior is correct. The assistant produced the code and report; the results shown are actual program outputs. Human understanding and any course submission remain the student's responsibility.
