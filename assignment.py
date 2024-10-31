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
    raise NotImplementedError


def solve_question_3(N: int, alpha: float, beta: float) -> float:
    """Grid world Markov decision process

    Args:
        N: Positive odd integer; size of the grid.
        alpha: Non-negative float; penalty parameter.
        beta: Non-negative float; penalty parameter.

    Returns:
        Total expected reward under an optimal policy.
    """
    raise NotImplementedError


#######################################################################################
# UTILITY FUNCTIONS
#######################################################################################


def init_matrix(
    *, nrows: int, ncols: int, fill_value: typing.Any = 0
) -> list[list[int]]:
    return [[fill_value] * ncols for _ in range(nrows)]


def n_choose_k(n: int, k: int) -> int:
    return math.comb(n, k)
