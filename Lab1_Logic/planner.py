import collections
from typing import Set, List, Optional, Tuple

class Action:
    """
    Represents an action in the planning domain.
    """
    def __init__(self, name: str, pos_pre: Set[str], neg_pre: Set[str], pos_eff: Set[str], neg_eff: Set[str]):
        """
        Initializes the action.
        
        :param name: Name of the action (e.g., 'Move(A, B)')
        :param pos_pre: Set of propositions that must be in the state
        :param neg_pre: Set of propositions that must NOT be in the state
        :param pos_eff: Set of propositions added to the state
        :param neg_eff: Set of propositions removed from the state
        """
        self.name = name
        self.pos_pre = pos_pre
        self.neg_pre = neg_pre
        self.pos_eff = pos_eff
        self.neg_eff = neg_eff

    def is_applicable(self, state: Set[str]) -> bool:
        """
        Checks if the action is applicable in the given state.
        An action is applicable if all positive preconditions are in the state
        and no negative preconditions are in the state.
        """
        # All positive preconditions must be present in the state
        if not self.pos_pre.issubset(state):
            return False
        # None of the negative preconditions can be present in the state
        if not self.neg_pre.isdisjoint(state):
            return False
        return True

    def apply(self, state: Set[str]) -> Set[str]:
        """
        Applies the action to the state and returns the new state.
        """
        new_state = state.copy()
        # 1. Remove negative effects
        new_state.difference_update(self.neg_eff)
        # 2. Add positive effects
        new_state.update(self.pos_eff)
        return new_state


def bfs_plan(initial_state: Set[str], goal: Set[str], actions: List[Action]) -> Optional[List[Action]]:
    """
    Uses Breadth-First Search to find a sequence of actions that reaches the goal state.
    
    :param initial_state: The starting state.
    :param goal: The propositions that must be true in the goal state.
    :param actions: List of all available actions.
    :return: A list of actions representing the plan, or None if no plan exists.
    """
    # A queue to hold the paths (sequences of actions) to explore
    # Each element is a tuple: (current_state, path_of_actions)
    queue = collections.deque([(initial_state, [])])
    
    # Keep track of visited states to avoid loops.
    # States are represented as frozensets so they can be hashed.
    visited = set([frozenset(initial_state)])

    while queue:
        current_state, path = queue.popleft()

        # Check if the goal is satisfied in the current state
        if goal.issubset(current_state):
            return path

        # Try to apply all possible actions
        for action in actions:
            if action.is_applicable(current_state):
                new_state = action.apply(current_state)
                new_state_frozen = frozenset(new_state)

                # Only explore the new state if it hasn't been visited yet
                if new_state_frozen not in visited:
                    visited.add(new_state_frozen)
                    queue.append((new_state, path + [action]))

    # If the queue is empty and the goal wasn't reached, no plan exists
    return None

def print_plan(initial_state: Set[str], plan: Optional[List[Action]]):
    """
    Utility function to print the plan and the states reached after each action.
    """
    if plan is None:
        print("No plan found.\n")
        return

    print("Plan found:")
    current_state = initial_state.copy()
    print(f"Initial State: {sorted(list(current_state))}")
    
    for i, action in enumerate(plan):
        print(f"Step {i+1}: {action.name}")
        current_state = action.apply(current_state)
        print(f"  State reached: {sorted(list(current_state))}")
    print()


def run_tests():
    """
    Runs the required tests on the generated planner for Task 3.
    """
    # ---------------------------------------------------------
    # Test A: Solvable Problem (Original Warehouse Problem)
    # ---------------------------------------------------------
    print("--- Test A: Solvable Problem ---")
    initial_state_A = {"At(Robot, A)", "At(Package, A)"}
    goal_A = {"At(Package, C)"}
    
    actions_A = [
        # Move actions
        Action("Move(A, B)", {"At(Robot, A)"}, set(), {"At(Robot, B)"}, {"At(Robot, A)"}),
        Action("Move(B, A)", {"At(Robot, B)"}, set(), {"At(Robot, A)"}, {"At(Robot, B)"}),
        Action("Move(B, C)", {"At(Robot, B)"}, set(), {"At(Robot, C)"}, {"At(Robot, B)"}),
        Action("Move(C, B)", {"At(Robot, C)"}, set(), {"At(Robot, B)"}, {"At(Robot, C)"}),
        
        # PickUp actions
        Action("PickUp(Package, A)", {"At(Robot, A)", "At(Package, A)"}, set(), {"Holding(Package)"}, {"At(Package, A)"}),
        Action("PickUp(Package, B)", {"At(Robot, B)", "At(Package, B)"}, set(), {"Holding(Package)"}, {"At(Package, B)"}),
        Action("PickUp(Package, C)", {"At(Robot, C)", "At(Package, C)"}, set(), {"Holding(Package)"}, {"At(Package, C)"}),
        
        # Drop actions
        Action("Drop(Package, A)", {"At(Robot, A)", "Holding(Package)"}, set(), {"At(Package, A)"}, {"Holding(Package)"}),
        Action("Drop(Package, B)", {"At(Robot, B)", "Holding(Package)"}, set(), {"At(Package, B)"}, {"Holding(Package)"}),
        Action("Drop(Package, C)", {"At(Robot, C)", "Holding(Package)"}, set(), {"At(Package, C)"}, {"Holding(Package)"}),
    ]
    
    plan_A = bfs_plan(initial_state_A, goal_A, actions_A)
    print_plan(initial_state_A, plan_A)

    # ---------------------------------------------------------
    # Test B: Impossible Problem
    # ---------------------------------------------------------
    print("--- Test B: Impossible Problem ---")
    print("Removing all PickUp actions so the robot cannot pick up the package.")
    initial_state_B = {"At(Robot, A)", "At(Package, A)"}
    goal_B = {"At(Package, C)"}
    
    # Filter out PickUp actions
    actions_B = [a for a in actions_A if not a.name.startswith("PickUp")]
    
    plan_B = bfs_plan(initial_state_B, goal_B, actions_B)
    print_plan(initial_state_B, plan_B)

    # ---------------------------------------------------------
    # Test C: Irrelevant Actions
    # ---------------------------------------------------------
    print("--- Test C: Irrelevant Actions ---")
    print("Robot can move to C without the package. The goal is At(Package, C).")
    initial_state_C = {"At(Robot, A)", "At(Package, A)"}
    goal_C = {"At(Package, C)"}
    
    # We will use actions_A since it contains all irrelevant moves by default.
    # The planner must not get confused and think moving the Robot to C satisfies At(Package, C).
    plan_C = bfs_plan(initial_state_C, goal_C, actions_A)
    print_plan(initial_state_C, plan_C)

if __name__ == "__main__":
    run_tests()
