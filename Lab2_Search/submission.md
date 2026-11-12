# Search and A* - Laboratory Submission

---

## 1. Task 0: Search Formulation

| Component | Your Specification |
| :--- | :--- |
| **State $S$** | The $(x, y)$ coordinate position of the robot in the 2D grid. |
| **Actions $A$** | $\{Up, Down, Left, Right\}$ |
| **Transition $T$** | $T((x,y), Up) = (x, y-1)$ (if $(x, y-1)$ is free `.` or `G`) |
| **Initial State $s_0$** | The $(x, y)$ coordinate marked with `S` |
| **Goal $G$** | The $(x, y)$ coordinate marked with `G` |
| **Cost $c$** | Constant cost of $1$ for every valid action |

**(a) What information is necessary to specify a state?**
Only the robot's current $(x, y)$ coordinates.

**(b) What makes an action invalid?**
An action is invalid if it moves the robot out of the map boundaries or into a cell containing an obstacle (`#`).

**(c) Is this a deterministic search problem?**
Yes. Every action has exactly one guaranteed outcome with no randomness.

**(d) What would constitute a solution?**
A sequence of valid actions (e.g., [Right, Right, Down, ...]) that successfully transitions the robot from the Initial State $s_0$ to the Goal State $G$.

---

## 2. Task 1 & 2: Agent Design and A* Generation Prompt

### Python Representation Design
- **State**: A Python tuple `(x, y)` representing grid coordinates.
- **Warehouse**: A 2D list or array of characters parsed from the ASCII string.
- **Valid Actions**: Determined by checking the map to ensure the neighboring `(x, y)` is within bounds and not equal to `#`.
- **Goal Recognition**: Checked when a state is popped from the frontier: `if current_state == goal_state`.
- **Frontier**: A priority queue (`heapq`) storing tuples of `(f_score, state)` to automatically expand the lowest cost node first.
- **Path Reconstruction**: A dictionary `came_from` mapping each state to the state that discovered it. 

### LLM Prompt Used
> "I am implementing a simple goal-based search agent in Python. The environment is a grid represented by an ASCII map. The agent starts at S and must reach G. The symbols `#` represent obstacles and `.` represents free cells. The agent can move up, down, left, or right, and every movement has cost 1. 
> Implement A* search. Use Manhattan distance as the heuristic: $h(n) = |x - x_G| + |y - y_G|$. 
> The program should represent grid positions as states, maintain a priority queue frontier, calculate $g(n)$, $h(n)$, and $f(n)$, avoid repeatedly expanding the same state using a visited list, reconstruct the path when the goal is reached, and return the path, path length, and number of states expanded. Write modular code and provide a test wrapper for multiple ASCII maps."

*(Note: No significant manual corrections were required for the generated code. The LLM correctly implemented `heapq` and the `came_from` dictionary for path reconstruction.)*

---

## 3. Task 3: Systematic Testing Results

| Test Case | Solution Found? | Path Length | States Expanded |
| :--- | :---: | :---: | :---: |
| **Test 1: Original map** | Yes | 40 | 63 |
| **Test 2: Trivial case** | Yes | 1 | 1 |
| **Test 3: No solution** | No | 0 | 9 |
| **Test 4: Alternative paths** | Yes | 6 | 14 |

---

## 4. Task 4: Inspect the A* Algorithm

| Concept | Where does it appear in the code? |
| :--- | :--- |
| **State** | `(x, y)` tuples stored in `current_state` and `next_state` variables. |
| **Action** | The coordinate offsets in `get_successors()`: `[(0, -1), (0, 1), (-1, 0), (1, 0)]` |
| **Transition** | `next_state = (x + dx, y + dy)` inside `get_successors()` |
| **Goal test** | `if current_state == problem.goal:` right after popping from the frontier. |
| **$g(n)$** | `tentative_g_score = g_score[current_state] + 1` |
| **$h(n)$** | `heuristic_func(next_state, problem.goal)` which calls `manhattan_distance()` |
| **$f(n)$** | `f_score = tentative_g_score + heuristic_func(...)` |
| **Frontier** | The `frontier = []` list managed by `heapq.heappush()` and `heapq.heappop()` |
| **Visited states** | Tracked implicitly via the `g_score` dictionary. If a state has a recorded `g_score`, it has been reached. |
| **Path reconstruction** | The `while current_state in came_from:` loop tracing backwards from the goal. |

**(a) What data structure is used for the A* frontier?** A priority queue (min-heap) implemented via Python's `heapq` module.
**(b) How does the program select the next state to expand?** By popping the element with the lowest $f(n)$ score from the priority queue.
**(c) Where is the heuristic calculated?** When a successor state is discovered, right before pushing it onto the frontier.
**(d) Does the program explicitly calculate $f(n) = g(n) + h(n)$?** Yes, exactly via the code `f_score = tentative_g_score + heuristic_func(...)`.
**(e) How does the program prevent unnecessary repeated exploration?** By keeping a dictionary of the lowest `g_score` for each state. If a state is reached again with a higher or equal cost, it is skipped.

---

## 5. Task 5: Compare A* with Blind Search

| Measure | BFS | A* |
| :--- | :---: | :---: |
| **Solution found** | Yes | Yes |
| **Path length** | 40 | 40 |
| **States expanded**| 63 | 63 |

**(a) Did both algorithms find a solution?** Yes.
**(b) Did they find paths of the same length?** Yes, both algorithms guarantee the optimal (shortest) path.
**(c) Which algorithm expanded fewer states?** In this specific maze, they expanded the exact same number of states!
**(d) Why might A* expand fewer states?** Normally, A* expands fewer states because the heuristic $h(n)$ aggressively pulls the search in the direction of the goal, avoiding dead-ends going backwards. However, in this specific tight, narrow maze full of obstacles, the reported counts happen to tie. Obstacles, heuristic values and tie-breaking all affect the explored states; the equal counts alone do not prove the maze has only one route.

---

## 6. Task 6: Investigate the Heuristic

**Why is Manhattan distance appropriate?**
With unit-cost horizontal and vertical moves, Manhattan distance is the exact shortest distance on an unobstructed grid. Obstacles may force a detour, so on this map it is a lower bound rather than necessarily the actual remaining cost.

**Experimental Results:**
1. **$h(n) = 0$**: Path Length = 40 | States Expanded = 63. *(Setting the heuristic to 0 degrades A* completely into BFS!)*
2. **Euclidean Distance**: Path Length = 40 | States Expanded = 63. *(Euclidean is the "straight line" distance. It is slightly less aggressive than Manhattan, and this run reports the same path length and expansion count).*
3. **Multiplied by 2 ($2 \times$ Manhattan)**: Path Length = 40 | States Expanded = 67. *(Doubling Manhattan distance can overestimate the true cost, removing the usual optimality guarantee. This run reports 67 expansions, but that number alone does not establish why they occurred. The implementation can pop an older heap entry after a better path was found, and its counter counts expansions rather than necessarily unique states.)*

---

## 7. Final Reflection

**1. Why is it important to formulate the search problem before writing the search algorithm?**
Because code is just syntax. If you don't mathematically define your state space, valid actions, and goal conditions on paper first, your code will be messy and logically flawed. Formulation separates the "what am I solving" from the "how do I type it."

**2. In what sense is A* an “informed” search algorithm?**
Unlike BFS, which blindly searches outward in expanding circles like spilled water, A* has "inside information"—a heuristic formula that points it toward the geographical coordinates of the goal, allowing it to stretch its search effort directly toward the target.

**3. Why does the choice of heuristic matter?**
The heuristic dictates the "personality" of the algorithm. If $h(n)$ is too weak (like $h(n)=0$), A* becomes slow and blind (BFS). Manhattan distance is admissible here, but obstacles can make it underestimate the remaining path; its efficiency depends on the map and tie-breaking. If $h(n)$ is too aggressive (overestimating), A* might skip the optimal path and return a bad, longer route.

**4. What did the LLM contribute to the engineering process?**
The LLM typed out the tedious standard boilerplate for A* (like managing the `heapq`, writing the path reconstruction `while` loop, and setting up the classes). I supplied the architectural design and rules; the LLM did the typing.

**5. What could go wrong if an engineer simply accepted LLM-generated code without testing it?**
The LLM might write an algorithm that looks like A* but has a subtle logical bug (e.g., forgetting to update the `came_from` dictionary for a cheaper path, or using an inadmissible heuristic). This would cause robots in production to take wildly inefficient routes or crash into walls, all while the code syntax technically looks "correct." Testing edge cases (like the "No Solution" map) is essential to catch these logic bugs.

## Task 3: Recorded path

Re-running the repository program produced the following path on the supplied warehouse map. Coordinates are `(x,y)`, with x increasing right and y down. Consecutive entries are adjacent; the 41 positions represent 40 moves.

```text
[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5), (13, 5), (13, 4), (13, 3), (12, 3), (11, 3), (10, 3), (9, 3), (8, 3), (7, 3), (7, 2), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7)]
```

The alternative-path test also returns six steps, equal to the Manhattan lower bound from `(1,1)` to `(5,3)`, establishing that this returned route is shortest for that test.

## Task 7: Evaluate the generated agent

1. **What worked:** The submitted program represents the grid, uses a heap frontier and reconstructs a route; its reported counts were reproduced for the four supplied tests.
2. **Problems and limits:** The A* loop does not skip stale heap entries, so its `states_expanded` counter can include the same state more than once. The doubled heuristic can overestimate, so its result is not generally guaranteed optimal. These are properties of the current code; the repository does not establish when they were first noticed.
3. **How checked:** Reading the heap loop and running the original, adjacent-goal, unreachable and alternative-path cases provides evidence for the stated behavior. A comparison with BFS on this map checks the returned length, not every possible map.
4. **Terminology and structures:** `heapq` is a min-heap: the smallest `(f,state)` tuple is removed first. `g_score` stores the best discovered path cost; `came_from` stores predecessors. Personal familiarity at the time of the original lab is not recorded.
5. **Changes:** The existing report says no significant manual corrections were required. The repository does not independently establish the full editing history, so no additional historical corrections are claimed here.
6. **Useful tests:** An unreachable goal checks termination; an adjacent goal checks the one-step boundary case; alternative paths check that the returned length matches a known lower bound.
7. **Trust:** Tests and code inspection are needed; a plausible route alone cannot establish general correctness.
8. **A* insight:** The priority `g+h` combines cost already paid with an estimate of remaining cost. An admissible heuristic preserves the usual shortest-path guarantee under appropriate implementation conditions; a larger heuristic may be faster or slower and may lose that guarantee.

The numerical counts above are from the submitted implementation and the particular maps; they should not be interpreted as independent evidence about all A* implementations.
