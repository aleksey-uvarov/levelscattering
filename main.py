import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize


def scatter_between_level_surfaces(x0: np.array, foo, levels,
                                   starting_direction=np.array([1, 0]),
                                   n_jumps=10):
    jump_points = np.zeros((x0.shape[0], n_jumps))
    current_direction = starting_direction
    current_point = x0
    for i in range(n_jumps):
        t = np.infty
        for level in levels:
            t_new = find_t_to_level(current_point, current_direction, foo, level)
            if t_new < t:
                t = t_new
        current_point = current_point + t * current_direction
        current_direction = next_direction(current_direction)
        jump_points[:, i] = current_point

    return jump_points


def find_t_to_level(x0, direction, foo, level):
    directional_difference = difference_along_path(x0, direction, foo, level)
    # t = optimize.broyden1(directional_difference, 0)
    t = optimize.newton(directional_difference, 1e-8)
    return t


def difference_along_path(x0: np.array, direction: np.array, foo, level):

    def f(t):
        return foo(x0 + t * direction) - level

    return f


def next_direction(direction):
    return direction[::-1]



if __name__ == "__main__":

    def f(x):
        return x[0]**2 + x[1]**2


    target_levels = [2, 3]
    x0 = np.array([0, 1])
    direction = np.array([1, 0])

    t = find_t_to_level(x0, direction, f, 2)
    print(t)
    print(x0 + t * direction)
    jump_points = scatter_between_level_surfaces(x0, f, target_levels)
    print(jump_points)

