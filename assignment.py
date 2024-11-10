import dataclasses
import math
import typing

#######################################################################################
# PART ONE
#   - You only need to implement the six "load" functions below.
#   - On Gradescope there will be two submissions:
#       1. Implement the six "load" functions and turn in this file.
#       2. Submit proofs that your test cases and expected results are correct.
#   - Do not change the file name or any function signatures.
#######################################################################################


@dataclasses.dataclass(frozen=True)
class Q1TestCase:
    pi_0: dict[int, float]
    n: int
    expected_result: dict[int, float]


@dataclasses.dataclass(frozen=True)
class Q2TestCase:
    n: int
    T: int
    S_0: int
    c: float
    expected_result: float


@dataclasses.dataclass(frozen=True)
class Q3TestCase:
    N: int
    alpha: float
    beta: float
    expected_result: float


def load_question_1_test_case_example() -> Q1TestCase:
    """Nothing for you to do here; this is what a test case should look like."""
    pi_0 = {2: 1.0}
    n = 2
    pi_n = {0: 1 / 3, 1: 5 / 24, 2: 5 / 24, 4: 1 / 4}
    return Q1TestCase(pi_0=pi_0, n=n, expected_result=pi_n)


def load_question_1_test_case_1() -> Q1TestCase:
    pi_0 = {1: 0.5, 3: 0.5}
    n = 2
    pi_n = {0: 31 / 96, 1: 9 / 32, 2: 7 / 96, 3: 19 / 96, 5: 1 / 8}
    return Q1TestCase(pi_0=pi_0, n=n, expected_result=pi_n)


def load_question_1_test_case_2() -> Q1TestCase:
    pi_0 = {0: 0.5, 1: 0.5}
    n = 1
    pi_n = {0: 1 / 2, 1: 1 / 4, 2: 1 / 4}
    return Q1TestCase(pi_0=pi_0, n=n, expected_result=pi_n)


def load_question_2_test_case_1() -> Q2TestCase:
    n = 1
    T = 1
    S_0 = 1
    c = 1
    expected_result = 2
    return Q2TestCase(n=n, T=T, S_0=S_0, c=c, expected_result=expected_result)


def load_question_2_test_case_2() -> Q2TestCase:
    n = 1
    T = 1
    S_0 = 3
    c = 2
    expected_result = 4
    return Q2TestCase(n=n, T=T, S_0=S_0, c=c, expected_result=expected_result)


# def load_question_3_test_case_1() -> Q3TestCase:
#     N =
#     alpha =
#     beta =
#     expected_result =
#     return Q3TestCase(N=N, alpha=alpha, beta=beta, expected_result=expected_result)


# def load_question_3_test_case_2() -> Q3TestCase:
#     N =
#     alpha =
#     beta =
#     expected_result =
#     return Q3TestCase(N=N, alpha=alpha, beta=beta, expected_result=expected_result)


#######################################################################################
# PART TWO
#   - You must implement the following three "solve" functions.
#   - The autograder will run them against a test suite.
#   - Do not change any of the function signatures.
#######################################################################################


def solve_question_1(pi_0: dict[int, float], n: int) -> dict[int, float]:
    """Markov chain on infinite state space.

    Args:
        pi_0: Dictionary mapping states to probabilities. You may assume all keys are
            non-negative integers, all values are non-negative, and the values sum
            to 1.
        n: Positive integer; the number of timesteps to simulate.

    Returns:
        A dictionary mapping states to probabilities. Any state not in the dictionary
        is assumed to have probability zero.
    """
    def incrementation_1(pi):
        pi_1 = {}
        for key in pi:
            if key == 0:
                pi_1[key] = pi_1.get(key, 0) + pi[key] * 1/2
                pi_1[key + 1] = pi_1.get(key + 1, 0) + pi[key] * 1/2
            else:
                for k in range(key):
                    pi_1[k] = pi_1.get(k, 0) + 1 / (2 * key) * pi[key]
                pi_1[key + 1] = pi_1.get(key + 1, 0) + 1 / 2 * pi[key]
        return pi_1
    pi_n = pi_0
    i = 0
    while i < n:
        print(i, pi_n)
        pi_n = incrementation_1(pi_n)
        i += 1
        print(i, pi_n)
        
    return pi_n

import math

def solve_question_2(n: int, T: int, S_0: int, c: float) -> float:
    """Finite-horizon Markov decision process.

    Args:
        n: Positive integer; the number of shares you must sell.
        T: Positive integer; finite time horizon.
        S_0: Positive integer; the initial stock price. You may assume S_0 > T + 1.
        c: Positive float; penalty parameter for over-selling.

    Returns:
        Total expected revenue under an optimal policy.
    """
    S = [S_0 + i for i in range(T + 1)]
    
    # Initialize V_T with precomputed values
    V_T = {i: S[-1] * i * 0.5 for i in range(2 * n + 1)}

    def binomial(i, n):
        return math.comb(n, i) * 0.5**i * (1 - 0.5)**(n - i)

    def reward(a, S_t, c, n):
        return sum(binomial(i, n) * (S_t * min(a, i) - c * max(0, a - i)) for i in range(n + 1))

    # Find optimal policy at each time step
    def find_optimal_policy(n, t, S, V_t):
        S_t = S[t]
        V_t_1 = {}
        
        # Calculate Q(s, a) values with optimized summing
        for s in range(2 * n + 1):
            Q = {}
            for a in range(s + 1):
                # print(sum_prob)
                Q[a] = reward(a, S_t, c, n) + sum(
                    binomial(j, n) * V_t[s - j]
                    for j in range(min(a + 1, n + 1))
                )
                if a < n:
                    Q[a] += sum(
                    binomial(j, n) * V_t[s - a]
                    for j in range(a + 1, n + 1)
                )
            V_t_1[s] = max(Q.values())
        return V_t_1

    # Backward induction to update V_T
    for t in range(T - 1, -1, -1):
        V_T = find_optimal_policy(n, t, S, V_T)

    return V_T[2 * n]


def solve_question_3(N: int, alpha: float, beta: float) -> float:
    """Grid world Markov decision process

    Args:
        N: Positive odd integer; size of the grid.
        alpha: Non-negative float; penalty parameter.
        beta: Non-negative float; penalty parameter.

    Returns:
        Total expected reward under an optimal policy.
    """
    # Initialize grid and parameters
    rewards = init_matrix(nrows=N, ncols=N)
    rewards[N-1][N-1] = 1    # Goal position reward
    rewards[N//2][N//2] = -alpha  # Penalty position
    rewards[N-1][0] = -beta      # Penalty position

    # Discount factor
    discount = 0.5
    
    # Initialize value function with zeros
    values = init_matrix(nrows=N, ncols=N, fill_value=0.0)
    threshold = 1e-6  # Convergence threshold

    # Possible actions
    actions = {
        'L': (0, -1),   # Left
        'R': (0, 1),    # Right
        'D': (1, 0),    # Down
        'U': (-1, 0)    # Up
    }

    # Value iteration loop
    while True:
        delta = 0  # Track maximum change for convergence
        new_values = [row[:] for row in values]  # Copy current values

        for i in range(N):
            for j in range(N):
                state_value = []
                
                # For each action, calculate expected value
                for _, (di, dj) in actions.items():
                    reward = rewards[i][j]  # Reward when leaving this state
                    expected_value = 0
                    
                    # Move in the intended direction first
                    new_i, new_j = i + di, j + dj
                    if new_i < 0 or new_i >= N:
                        new_i = i  # Stay in place if hitting a wall

                    if new_j < 0 or new_j >= N:
                        new_j = j  # Stay in place if hitting a wall

                    # After moving in the intended direction, consider "blown" movements
                    
                    # 50% chance to stay in the new position
                    expected_value += 0.5 * (reward + discount * values[new_i][new_j])
                    
                    # 25% chance to be blown left
                    blown_left_i, blown_left_j = new_i, max(0, new_j - 1)
                    expected_value += 0.25 * (reward + discount * values[blown_left_i][blown_left_j])
                    
                    # 25% chance to be blown down
                    blown_down_i, blown_down_j = min(N - 1, new_i + 1), new_j
                    expected_value += 0.25 * (reward + discount * values[blown_down_i][blown_down_j])

                    state_value.append(expected_value)

                # Update the value for state (i, j) with the max value of all actions
                new_values[i][j] = max(state_value)
                delta = max(delta, abs(new_values[i][j] - values[i][j]))

        values = new_values  # Update values
        if delta < threshold:
            break  # Converged

    # Return the total expected reward starting from the initial state (0,0)
    return values[0][0]


#######################################################################################
# UTILITY FUNCTIONS
#######################################################################################


def init_matrix(
    *, nrows: int, ncols: int, fill_value: typing.Any = 0
) -> list[list[int]]:
    return [[fill_value] * ncols for _ in range(nrows)]


def n_choose_k(n: int, k: int) -> int:
    return math.comb(n, k)
