import heapq
import collections
import time

WAREHOUSE_MAP_1 = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
""".strip()

WAREHOUSE_MAP_TRIVIAL = """
#####
#SG##
#####
""".strip()

WAREHOUSE_MAP_NO_SOLUTION = """
#######
#S....#
###.###
#...#G#
#######
""".strip()

WAREHOUSE_MAP_ALTERNATIVE = """
#######
#S....#
#.....#
#....G#
#######
""".strip()

class SearchProblem:
    def __init__(self, ascii_map):
        self.grid = ascii_map.strip().split('\n')
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0
        self.start = None
        self.goal = None
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 'S':
                    self.start = (x, y)
                elif self.grid[y][x] == 'G':
                    self.goal = (x, y)

    def is_valid_state(self, state):
        x, y = state
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] != '#'
        return False

    def get_successors(self, state):
        x, y = state
        successors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Up, Down, Left, Right
            next_state = (x + dx, y + dy)
            if self.is_valid_state(next_state):
                successors.append(next_state)
        return successors

def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

def euclidean_distance(state1, state2):
    return ((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2) ** 0.5

def heuristic_zero(state1, state2):
    return 0

def heuristic_double_manhattan(state1, state2):
    return 2 * manhattan_distance(state1, state2)

def a_star_search(problem, heuristic_func=manhattan_distance):
    if not problem.start or not problem.goal:
        return None, 0
    
    frontier = []
    # heapq format: (f_score, state)
    heapq.heappush(frontier, (0, problem.start))
    
    came_from = {}
    g_score = {problem.start: 0}
    states_expanded = 0
    
    while frontier:
        current_f, current_state = heapq.heappop(frontier)
        
        # If we reached the goal, reconstruct path
        if current_state == problem.goal:
            path = []
            while current_state in came_from:
                path.append(current_state)
                current_state = came_from[current_state]
            path.append(problem.start)
            path.reverse()
            return path, states_expanded
            
        states_expanded += 1
        
        for next_state in problem.get_successors(current_state):
            tentative_g_score = g_score[current_state] + 1 # Cost is always 1
            
            if next_state not in g_score or tentative_g_score < g_score[next_state]:
                came_from[next_state] = current_state
                g_score[next_state] = tentative_g_score
                f_score = tentative_g_score + heuristic_func(next_state, problem.goal)
                heapq.heappush(frontier, (f_score, next_state))
                
    return None, states_expanded

def bfs_search(problem):
    if not problem.start or not problem.goal:
        return None, 0
        
    frontier = collections.deque([problem.start])
    came_from = {problem.start: None}
    states_expanded = 0
    
    while frontier:
        current_state = frontier.popleft()
        
        if current_state == problem.goal:
            path = []
            while current_state is not None:
                path.append(current_state)
                current_state = came_from[current_state]
            path.reverse()
            return path, states_expanded
            
        states_expanded += 1
        
        for next_state in problem.get_successors(current_state):
            if next_state not in came_from:
                came_from[next_state] = current_state
                frontier.append(next_state)
                
    return None, states_expanded

def print_result(name, path, expanded):
    if path:
        print(f"[{name}] Solution found! Length: {len(path)-1}, States Expanded: {expanded}")
    else:
        print(f"[{name}] No solution found. States Expanded: {expanded}")

if __name__ == "__main__":
    print("=== TASK 3: SYSTEMATIC TESTING (A*) ===")
    p1 = SearchProblem(WAREHOUSE_MAP_1)
    p2 = SearchProblem(WAREHOUSE_MAP_TRIVIAL)
    p3 = SearchProblem(WAREHOUSE_MAP_NO_SOLUTION)
    p4 = SearchProblem(WAREHOUSE_MAP_ALTERNATIVE)
    
    path, exp = a_star_search(p1)
    print_result("Test 1: Original Map", path, exp)
    
    path, exp = a_star_search(p2)
    print_result("Test 2: Trivial Map", path, exp)
    
    path, exp = a_star_search(p3)
    print_result("Test 3: No Solution", path, exp)
    
    path, exp = a_star_search(p4)
    print_result("Test 4: Alternative Paths", path, exp)
    
    print("\n=== TASK 5: BFS vs A* COMPARISON ===")
    path_bfs, exp_bfs = bfs_search(p1)
    path_astar, exp_astar = a_star_search(p1)
    print(f"BFS: Path Length={len(path_bfs)-1 if path_bfs else 'None'}, Expanded={exp_bfs}")
    print(f"A*:  Path Length={len(path_astar)-1 if path_astar else 'None'}, Expanded={exp_astar}")
    
    print("\n=== TASK 6: HEURISTIC INVESTIGATION ===")
    path_h0, exp_h0 = a_star_search(p1, heuristic_zero)
    print_result("Heuristic: h(n) = 0", path_h0, exp_h0)
    
    path_euc, exp_euc = a_star_search(p1, euclidean_distance)
    print_result("Heuristic: Euclidean", path_euc, exp_euc)
    
    path_2m, exp_2m = a_star_search(p1, heuristic_double_manhattan)
    print_result("Heuristic: 2 * Manhattan", path_2m, exp_2m)
