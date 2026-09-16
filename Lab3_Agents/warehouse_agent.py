import heapq

WAREHOUSE_MAP = """
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
""".strip()

class GoalBasedAgent:
    """
    A Goal-Based Agent that uses A* search to formulate a plan
    before acting in the environment.
    """
    def __init__(self, ascii_map):
        self.grid = ascii_map.strip().split('\n')
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0
        self.start = None
        self.goal = None
        
        # Parse the environment to find initial state and goal
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 'S':
                    self.start = (x, y)
                elif self.grid[y][x] == 'G':
                    self.goal = (x, y)

    def is_free(self, x, y):
        """Sensor: Checks if a coordinate is free space."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] != '#'
        return False

    def get_valid_actions(self, state):
        """Actuators: Returns a list of valid states the agent can move to."""
        x, y = state
        actions = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Up, Down, Left, Right
            if self.is_free(x + dx, y + dy):
                actions.append((x + dx, y + dy))
        return actions

    def heuristic(self, state):
        """Manhattan distance heuristic to guide the search."""
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def formulate_plan(self):
        """
        Decision Making Component:
        Uses A* Search to find the optimal collision-free path to the goal.
        """
        print("Agent is formulating a plan...")
        if not self.start or not self.goal:
            return None
            
        frontier = []
        heapq.heappush(frontier, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == self.goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                path.reverse()
                return path
                
            for nxt in self.get_valid_actions(current):
                tentative_g_score = g_score[current] + 1
                if nxt not in g_score or tentative_g_score < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(nxt)
                    heapq.heappush(frontier, (f_score, nxt))
                    
        return None

if __name__ == "__main__":
    agent = GoalBasedAgent(WAREHOUSE_MAP)
    path = agent.formulate_plan()
    
    if path:
        print(f"Plan formulated successfully! Path length: {len(path)-1} steps.")
        print(f"Execution sequence: {path}")
    else:
        print("Error: No valid path exists to the goal!")
