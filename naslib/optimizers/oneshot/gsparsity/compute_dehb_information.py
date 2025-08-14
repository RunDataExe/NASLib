import math


def compute_dehb_params(min_resource, max_resource, reduction_factor):
    # Compute s_max and n_brackets
    s_max = int(math.floor(math.log(max_resource / min_resource, reduction_factor)))
    n_brackets = s_max + 1

    # Budget candidates
    budget_candidates = [min_resource * reduction_factor**i for i in range(n_brackets)]

    # n_trials_for_each_bracket
    n_trials_for_each_bracket = []
    for i in range(n_brackets):
        n = 0
        for j in range(i, n_brackets):
            n += reduction_factor ** (s_max - j)
        n_trials_for_each_bracket.append(n)

    # n_trials_in_the_bracket
    n_trials_in_the_bracket = []
    for i in range(n_brackets):
        n_trials = []
        for j in range(i, n_brackets):
            n = reduction_factor ** (s_max - j)
            n_trials.append(n)
        n_trials_in_the_bracket.append(n_trials)

    # Total trials per iteration
    n_trials_per_iteration = sum(n_trials_for_each_bracket)

    print(f"s_max: {s_max}")
    print(f"n_brackets: {n_brackets}")
    print(f"n_trials_per_iteration: {n_trials_per_iteration}")
    print("budget_candidates:", budget_candidates)
    print("n_trials_for_each_bracket:", n_trials_for_each_bracket)
    print("n_trials_in_the_bracket:")
    for i, bracket in enumerate(n_trials_in_the_bracket):
        print(f"  Bracket {i}: {bracket}")
    print("-" * 40)


# Example usage:
print("Oneshot 70 setting:")
compute_dehb_params(min_resource=1, max_resource=70, reduction_factor=3)

print("Oneshot 27:")
compute_dehb_params(min_resource=1, max_resource=27, reduction_factor=3)


print("Oneshot 81:")
compute_dehb_params(min_resource=1, max_resource=81, reduction_factor=3)

print("Two Phase setting:")
compute_dehb_params(min_resource=21, max_resource=70, reduction_factor=3)


print("Two Phase test:")
compute_dehb_params(min_resource=21, max_resource=567, reduction_factor=3)

print("Oneshot very low budget setting:")
compute_dehb_params(min_resource=1, max_resource=3, reduction_factor=3)

print("Oneshot bit lower budget setting:")
compute_dehb_params(min_resource=1, max_resource=9, reduction_factor=3)
