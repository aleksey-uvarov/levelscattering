import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.patches import Rectangle


def scatter_between_level_surfaces(x0: np.array, foo, levels,
                                   starting_direction=np.array([1, 0]),
                                   n_jumps=4):
    """Returns the coordinates of cusp points in a line that
    is formed by straight segments connecting the level surfaces of foo.
    The output contains (n_jumps + 1) points. The 0th point is taken by going upwards
    from the first point.
    We assume that foo is monotonously increasing (non strictly) in all arguments.
    """
    jump_points = np.zeros((x0.shape[0], n_jumps + 2))

    # find the first intersection
    level = min(levels)
    t = find_t_to_level(x0, starting_direction, foo, level)
    first_intersection = x0 + t * starting_direction
    jump_points[:, 1] = first_intersection

    # find the zeroth point
    t = find_t_to_level(first_intersection,
                        next_direction(starting_direction),
                        foo, max(levels))
    jump_points[:, 0] = first_intersection + t * next_direction(starting_direction)

    # find the remaining points
    current_point = first_intersection
    current_direction = starting_direction
    for i in range(n_jumps):
        if i % 2 == 0:
            level = max(levels)
        else:
            level = min(levels)
        t = find_t_to_level(current_point, current_direction, foo, level)
        current_point = current_point + t * current_direction
        current_direction = next_direction(current_direction)
        jump_points[:, i + 2] = current_point

    return jump_points


def find_t_to_level(x_start, line_direction, foo, level):
    directional_difference = difference_along_path(x_start, line_direction, foo, level)
    t = optimize.broyden1(directional_difference, 0)
    # t = optimize.newton(directional_difference, 1e-8)
    return t


def difference_along_path(x0: np.array, direction: np.array, foo, level):

    def f(t):
        return - foo(x0 + t * direction) + level

    return f


def next_direction(direction):
    new_direction = np.zeros_like(direction)
    new_direction[0] = direction[1]
    new_direction[1] = direction[0]
    return new_direction


def three_points_to_rectangle_params(a, b, c):
    """Takes three corners of a rectangle and
    returns rectangle parameters for matplotlib (xy, width, height)"""
    x_low = min([a[0], b[0], c[0]])
    y_low = min([a[1], b[1], c[1]])
    x_high = max([a[0], b[0], c[0]])
    y_high = max([a[1], b[1], c[1]])
    xy = (x_low, y_low)
    width = x_high - x_low
    height = y_high - y_low
    return xy, width, height


def draw_rectangles_on_axis(jump_points, ax):
    for i in range(jump_points.shape[1] - 2):
        xy, width, height = three_points_to_rectangle_params(
            jump_points[:, i], jump_points[:, i+1], jump_points[:, i+2])
        ax.add_patch(Rectangle(xy, width, height, alpha=0.3, color='tab:blue'))


if __name__ == "__main__":

    def f(x):
            return x[0] * x[1]


    target_levels = [2, 3]
    x0 = np.array([0.1, 5])
    direction = np.array([1, 0])

    # t = find_t_to_level(x0, direction, f, 2)
    # print(t)
    # print(x0 + t * direction)
    the_jump_points = scatter_between_level_surfaces(x0, f, target_levels, n_jumps=5)

    fig, ax = plt.subplots()
    draw_rectangles_on_axis(the_jump_points, ax)
    xs = np.linspace(min(the_jump_points[0, :]), max(the_jump_points[0, :]), num=101)
    plt.scatter(x0[0], x0[1])
    plt.plot(xs, target_levels[0] / xs, color='black')
    plt.plot(xs, target_levels[1] / xs, color='black')
    plt.grid()
    plt.plot(the_jump_points[0, :], the_jump_points[1, :], color='red')
    plt.show()

