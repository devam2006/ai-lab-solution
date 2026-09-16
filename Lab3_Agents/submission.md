# Goal-Based Agents - Laboratory Submission

---

## 1. Task 1: Understanding the Problem

**1. What is the environment?**
The environment is the 2D grid warehouse, defined by the layout of free spaces (`.`) and immovable shelves/obstacles (`#`).

**2. What is the goal of the agent?**
To physically navigate its location from the starting loading bay (`S`) to the dispatch area (`G`).

**3. What actions are available to the agent?**
The agent can move exactly one grid square in four cardinal directions: `Up`, `Down`, `Left`, and `Right`.

**4. What information must the agent maintain in order to choose its next action?**
The agent must keep track of its current $(x, y)$ coordinate position on the map, the known locations of the obstacles, and the calculated path it plans to take to reach the goal.

**5. Why is this an example of a goal-based agent rather than a simple reflex agent?**
A simple reflex agent just reacts to what is immediately in front of it (e.g., "if wall in front, turn right"). This would lead to getting permanently stuck in a U-shaped dead end. A **goal-based agent** "thinks before it acts"—it considers the future, calculates an entire path (a sequence of actions) to achieve its explicit goal, and *then* executes those actions.

**Think About It: What if the warehouse becomes twice as large?**
If the warehouse scales up, the same search strategy (like A*) would technically still work. However, the *difficulty* arises in computational cost. A larger grid means an exponentially larger number of possible paths. An uninformed search (like BFS) would become too slow, meaning a very strong heuristic would become strictly necessary to solve the map in a reasonable amount of time.

---

## 2. Task 2: Designing the Agent

- **Environment**: A discrete, deterministic, fully observable 2D grid mapping free spaces and obstacles.
- **Current State**: The agent's $(x, y)$ coordinates.
- **Goal**: Reach coordinate $(x_G, y_G)$.
- **Available Actions**: $Move(0, -1)$, $Move(0, 1)$, $Move(-1, 0)$, $Move(1, 0)$.
- **Decision-Making Component**: A pathfinding algorithm (A*) that takes the map, start, and goal, and outputs an optimal sequence of movements.

**Agent Architecture Flow:**
`Environment Map & Goal` $\rightarrow$ `Decision Component (A* Search)` $\rightarrow$ `Action Sequence Generated` $\rightarrow$ `Actuators execute moves in Environment`

---

## 3. Task 3: Prompt Engineering

**Prompt Used:**
> "Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem. The program should:
> - represent the warehouse as a two-dimensional grid;
> - determine a collision-free path from S to G;
> - avoid all obstacles;
> - print either the path found or a suitable message if no path exists;
> - use the A* search algorithm with Manhattan distance and explain why it was chosen."

**1. Did the LLM generate a working program on the first attempt?**
Yes. The LLM accurately translated the requirements into a working object-oriented Python script.

**2. If not, how can you improve your prompt?**
N/A (Though if it failed, breaking the prompt down to explicitly define what a "state tuple" is would help).

**3. What search algorithm did the LLM choose?**
It explicitly implemented A* Search, using the `heapq` module to manage the frontier.

**4. Why do you think the LLM selected this algorithm?**
A* is the industry-standard algorithm for 2D grid pathfinding. It was selected because it guarantees the shortest path (optimal) while exploring far fewer states than Breadth-First Search, making it highly efficient.

---

## 4. Final Reflection: Responsible Use of LLMs

**The LLM's Contribution to Engineering:**
The LLM acted as a rapid prototyper. It handled the tedious task of setting up Python classes, writing the string-parsing logic to convert the ASCII map into a 2D array, and writing the boilerplate priority queue loop for A*. 

**The Danger of Blind Acceptance:**
An engineer must never blindly accept LLM code because the model is capable of generating "plausible but logically fatal" algorithms. For example, the LLM might accidentally implement a heuristic that is mathematically *inadmissible* (meaning it overestimates the cost). The code would run perfectly without crashing, and it would return a path, but the path would **not be optimal**. Without independent testing and verification of the logic, the engineer would deploy a permanently flawed, inefficient robot into the real warehouse!

---
**Core Takeaway:** 
$AI Science \rightarrow AI Engineering$
The science is formulating the math and the rules. The engineering is writing the code. We can use LLMs to automate the engineering, but the human must remain entirely responsible for verifying the science.
