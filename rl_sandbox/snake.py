from __future__ import annotations

import queue
import random
import tkinter

from enum import auto, StrEnum
from typing import Iterable, Sequence

import torch
import torch.nn
import torch.nn.functional


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


class PlayingState:
    width: int
    height: int
    apple_position: Vec2i
    snake_direction: Vec2i
    snake_positions: list[Vec2i]
    prng: random.Random

    class Action(StrEnum):
        UP = auto()
        DOWN = auto()
        LEFT = auto()
        RIGHT = auto()
        IDLE = auto()

    def __init__(
        self,
        *,
        width: int,
        height: int,
        apple_position: Vec2i,
        snake_direction: Vec2i,
        snake_positions: list[Vec2i],
        prng: random.Random
    ) -> None:
        self.width = width
        self.height = height
        self.apple_position = apple_position
        self.snake_direction = snake_direction
        self.snake_positions = snake_positions
        self.prng = prng

    def step(self, action: Action) -> State:
        width = self.width
        height = self.height
        current_apple_position = self.apple_position
        current_snake_direction = self.snake_direction
        current_snake_positions = self.snake_positions
        prng = self.prng
        bounds = ((0, 0), (width, height))
        directions = {
            self.Action.UP: (0, -1),
            self.Action.DOWN: (0, 1),
            self.Action.LEFT: (-1, 0),
            self.Action.RIGHT: (1, 0),
            self.Action.IDLE: current_snake_direction
        }

        updated_snake_direction = directions[action]

        snake_direction_diff = vec2i_add(
            current_snake_direction,
            updated_snake_direction
        )

        if snake_direction_diff == (0, 0):
            updated_snake_direction = current_snake_direction

        current_snake_head = current_snake_positions[0]
        updated_snake_head = vec2i_add(
            current_snake_head,
            updated_snake_direction
        )

        if not check_bounds(bounds, updated_snake_head):
            return LostState()

        if updated_snake_head == current_apple_position:
            updated_snake_positions = \
                [updated_snake_head] + current_snake_positions

            if len(updated_snake_positions) == (width * height):
                return WonState()

            updated_apple_position = random_vec2i(
                bounds,
                exclude=updated_snake_positions,
                prng=prng
            )
        else:
            updated_snake_positions = \
                [updated_snake_head] + current_snake_positions[:-1]

            updated_apple_position = current_apple_position

        if check_duplicity(updated_snake_positions):
            return LostState()

        return PlayingState(
            width=width,
            height=height,
            apple_position=updated_apple_position,
            snake_direction=updated_snake_direction,
            snake_positions=updated_snake_positions,
            prng=prng
        )

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


type State = Union[PlayingState, LostState, WonState]


class App:
    def __init__(self, root, width, height, size):
        self.root = root
        self.width = width
        self.height = height
        self.size = size
        self.state = PlayingState.random_start(width=width, height=height)
        self.actions = queue.Queue()
        self.canvas = tkinter.Canvas(
            root,
            width=width * size,
            height=height * size,
            bg="black"
        )

        self.canvas.pack(padx=10, pady=10)

    def start(self):
        def on_key_press(event):
            actions = {
                "Up": PlayingState.Action.UP,
                "Down": PlayingState.Action.DOWN,
                "Left": PlayingState.Action.LEFT,
                "Right": PlayingState.Action.RIGHT
            }

            key = event.keysym

            if key in actions:
                self.actions.put(actions[key])

        self.root.bind("<KeyPress>", on_key_press)
        self.draw()
        self.update()

    def update(self):
        if not isinstance(self.state, PlayingState):
            return

        try:
            action = self.actions.get(False)
        except queue.Empty:
            action = PlayingState.Action.IDLE

        self.state = self.state.step(action)

        self.root.after(100, self.update)

    def draw(self):
        self.canvas.delete("all")

        if not isinstance(self.state, PlayingState):
            return

        for y in range(self.height):
            for x in range(self.width):
                position = (x, y)

                if position == self.state.apple_position:
                    self.canvas.create_rectangle(
                        x * self.size,
                        y * self.size,
                        (x * self.size) + self.size,
                        (y * self.size) + self.size,
                        fill='red'
                    )
                elif position in self.state.snake_positions:
                    self.canvas.create_rectangle(
                        x * self.size,
                        y * self.size,
                        (x * self.size) + self.size,
                        (y * self.size) + self.size,
                        fill='green'
                    )

        self.root.after(int(1000 / 30), self.draw)


def iterate_policy(
    initial_state: PlayingState,
    policy: torch.nn.Sequential
) -> Iterable[
    tuple[
        PlayingState,
        PlayingState.Action,
        float,
        list[torch.Tensor]
    ]
]:
    # def features(state: PlayingState) -> torch.Tensor:
    #     grid = []

    #     for y in range(state.height):
    #         for x in range(state.width):
    #             position = (x, y)

    #             value = None

    #             if position == state.apple_position:
    #                 value = 2.0
    #             elif position in state.snake_positions:
    #                 value = 1.0
    #             else:
    #                 value = 0.0

    #             grid.append(value)

    #     return torch.tensor(grid, dtype=torch.float32)

    def features(state: PlayingState) -> torch.Tensor:
        head_x, head_y = state.snake_positions[0]
        apple_x, apple_y = state.apple_position
        dx = apple_x - head_x
        dy = apple_y - head_y
        danger_up = 1.0 if (head_y - 1 < 0 or (head_x, head_y - 1) in state.snake_positions) else 0.0
        danger_down = 1.0 if (head_y + 1 >= state.height or (head_x, head_y + 1) in state.snake_positions) else 0.0
        danger_left = 1.0 if (head_x - 1 < 0 or (head_x - 1, head_y) in state.snake_positions) else 0.0
        danger_right = 1.0 if (head_x + 1 >= state.width or (head_x + 1, head_y) in state.snake_positions) else 0.0
        return torch.tensor([head_x, head_y, dx, dy, *state.snake_direction], dtype=torch.float32)

    def step(
        current_state: PlayingState,
        action: PlayingState.Action
    ) -> tuple[State, float]:
        updated_state = current_state.step(action)

        if isinstance(updated_state, LostState):
            return updated_state, -1.0

        if isinstance(updated_state, WonState):
            return updated_state, 100.0

        head_x, head_y = updated_state.snake_positions[0]
        apple_x, apple_y = updated_state.apple_position
        dx = apple_x - head_x
        dy = apple_y - head_y

        if current_state.apple_position != updated_state.apple_position:
            return updated_state, 10.0

        import math

        return updated_state, -math.sqrt(dx ** 2 + dy ** 2)

    current_state = initial_state

    while isinstance(current_state, PlayingState):
        x = features(current_state)
        y = policy(x)
        action = torch.multinomial(y, 1).item()
        z = torch.log(y[action])
        z.backward()
        gradients = [
            parameter.grad.clone()
            for parameter in policy.parameters()
        ]

        policy.zero_grad()

        updated_state, reward = step(
            current_state,
            list(PlayingState.Action)[action]
        )

        yield current_state, action, reward, gradients

        current_state = updated_state


def evaluate_policy(
    initial_state: PlayingState,
    policy: torch.nn.Sequential,
    gamma: float
) -> list[
    tuple[
        PlayingState,
        PlayingState.Action,
        float,
        float,
        list[torch.Tensor]
    ]
]:
    steps = list(iterate_policy(initial_state, policy))
    result = []

    for i, (state, action, reward, gradients) in enumerate(steps):
        returns = sum(
            (gamma ** j) * r for j, (_, _, r, _) in enumerate(steps[i:])
        )

        item = (state, action, reward, returns, gradients)
        result.append(item)

    return result


def draw_state(canvas, state, size):
    canvas.delete("all")

    if not isinstance(state, PlayingState):
        return

    for y in range(state.height):
        for x in range(state.width):
            position = (x, y)

            if position == state.apple_position:
                canvas.create_rectangle(
                    x * size,
                    y * size,
                    (x * size) + size,
                    (y * size) + size,
                    fill='red'
                )
            elif position in state.snake_positions:
                canvas.create_rectangle(
                    x * size,
                    y * size,
                    (x * size) + size,
                    (y * size) + size,
                    fill='green'
                )


def learn_reinforce(
    *,
    width: int = 10,
    height: int = 10,
    episodes: int = 10000,
    alpha: float = 0.00025,
    gamma: float = 0.99
) -> None:
    policy = torch.nn.Sequential(
        torch.nn.Linear(6, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, len(PlayingState.Action)),
        torch.nn.Softmax(dim=-1)
    )

    progress = []

    for _ in range(episodes):
        initial_state = PlayingState.random_start(width=width, height=height)
        steps = evaluate_policy(initial_state, policy, gamma)

        for t, (_, _, _, returns, gradients) in enumerate(steps):
            for parameter, gradient in zip(policy.parameters(), gradients):
                #print(alpha * (gamma ** t) * returns * gradient)
                #print(torch.mean(parameter.data))
                #exit()
                parameter.data += alpha * (gamma ** t) * returns * gradient

        progress.append(steps[0][3])

    import matplotlib.pyplot as plt

    plt.plot(progress, ".")
    plt.show()

    import tkinter as tk

    root = tk.Tk()
    root.title("Snake Game")
    
    states = evaluate_policy(PlayingState.random_start(width=width, height=height), policy, gamma)

    canvas = tk.Canvas(
        root,
        width=width * 20,
        height=height * 20,
        bg='white'
    )

    canvas.pack()
    
    def animate(index=0):
        if index < len(states):
            draw_state(canvas, states[index][0], 20)
            root.after(int(0.5 * 1000), lambda: animate(index + 1))

    animate()
    root.mainloop()

    return progress


learn_reinforce()
