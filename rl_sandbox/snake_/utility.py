import random

from typing import Sequence


type Vec2i = tuple[int, int]
type Rect2i = tuple[Vec2i, Vec2i]


def vec2i_add(a: Vec2i, b: Vec2i) -> Vec2i:
    return (a[0] + b[0], a[1] + b[1])


def vec2i_sub(a: Vec2i, b: Vec2i) -> Vec2i:
    return (a[0] - b[0], a[1] - b[1])


def vec2i_neg(a: Vec2i) -> Vec2i:
    return (-a[0], -a[1])


def random_vec2i(
    bounds: Rect2i,
    exclude: Sequence[Vec2i] = [],
    prng: random.Random = random
) -> Vec2i:
    x = prng.randrange(bounds[0][0], bounds[1][0])
    y = prng.randrange(bounds[0][1], bounds[1][1])

    while (x, y) in exclude:
        x += 1

        if x >= bounds[1][0]:
            x = bounds[0][0]
            y += 1

            if y >= bounds[1][1]:
                y = bounds[0][1]

    return (x, y)


def check_bounds(bounds: Rect2i, a: Vec2i) -> bool:
    if a[0] < bounds[0][0] or a[0] >= bounds[1][0]:
        return False

    if a[1] < bounds[0][1] or a[1] >= bounds[1][1]:
        return False

    return True


def check_duplicity[T](items: Sequence[T]) -> bool:
    return len(items) != len(set(items))
