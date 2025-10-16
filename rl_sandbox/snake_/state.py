import random

from .utility import (
    Vec2i,
    check_bounds,
    check_duplicity,
    random_vec2i,
    vec2i_add,
    vec2i_neg,
    vec2i_sub
)


class Snake:
    direction: Vec2i
    positions: list[Vec2i]

    def __init__(self, direction: Vec2i, positions: list[Vec2i]) -> None:
        self.direction = direction
        self.positions = positions


class SnakeAction:
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class PlayingState:
    width: int
    height: int
    apple_position: Vec2i
    snakes: dict[str, Snake]
    prng: random.Random

    def __init__(
        self,
        *,
        width: int,
        height: int,
        apple_position: Vec2i,
        snakes: dict[str, Snake],
        prng: random.Random
    ) -> None:
        self.width = width
        self.height = height
        self.apple_position = apple_position
        self.snakes = snakes
        self.prng = prng

    def update(self, actions: dict[str, SnakeAction]) -> State:
        width = self.width
        height = self.height
        current_apple_position = self.apple_position
        current_snakes = self.snakes
        prng = self.prng
        bounds = ((0, 0), (width, height))

        updated_snakes = {}

        for snake_id, snake in current_snakes.items():


        # updated_snake_direction = directions[action]

        # snake_direction_diff = vec2i_add(
        #     current_snake_direction,
        #     updated_snake_direction
        # )

        # if snake_direction_diff == (0, 0):
        #     updated_snake_direction = current_snake_direction

        # current_snake_head = current_snake_positions[0]
        # updated_snake_head = vec2i_add(
        #     current_snake_head,
        #     updated_snake_direction
        # )

        # if not check_bounds(bounds, updated_snake_head):
        #     return LostState()

        # if updated_snake_head == current_apple_position:
        #     updated_snake_positions = \
        #         [updated_snake_head] + current_snake_positions

        #     if len(updated_snake_positions) == (width * height):
        #         return WonState()

        #     updated_apple_position = random_vec2i(
        #         bounds,
        #         exclude=updated_snake_positions,
        #         prng=prng
        #     )
        # else:
        #     updated_snake_positions = \
        #         [updated_snake_head] + current_snake_positions[:-1]

        #     updated_apple_position = current_apple_position

        # if check_duplicity(updated_snake_positions):
        #     return LostState()

        # return PlayingState(
        #     width=width,
        #     height=height,
        #     apple_position=updated_apple_position,
        #     snake_direction=updated_snake_direction,
        #     snake_positions=updated_snake_positions,
        #     prng=prng
        # )

    @classmethod
    def random_start(
        cls,
        *,
        width: int,
        height: int,
        prng: random.Random = random
    ):
        bounds = ((0, 0), (width, height))

        while True:
            snake_head_position = random_vec2i(bounds, prng=prng)
            snake_direction = prng.choice([
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1)
            ])

            snake_body_position = vec2i_add(
                snake_head_position,
                vec2i_neg(snake_direction)
            )

            forward_clear = check_bounds(
                bounds,
                vec2i_add(
                    snake_head_position,
                    snake_direction
                )
            )

            if not forward_clear:
                continue

            backward_clear = check_bounds(bounds, snake_body_position)

            if not backward_clear:
                continue

            snake_positions = [snake_head_position, snake_body_position]
            apple_position = random_vec2i(
                bounds,
                exclude=snake_positions,
                prng=prng
            )

            return PlayingState(
                width=width,
                height=height,
                apple_position=apple_position,
                snake_direction=snake_direction,
                snake_positions=snake_positions,
                prng=prng
            )


class LostState:
    def step(self, action) -> State:
        raise TypeError


class WonState:
    def step(self, action) -> State:
        raise TypeError


type State = Union[LostState, PlayingState, WonState]
