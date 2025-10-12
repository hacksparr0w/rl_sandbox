from __future__ import annotations

import queue
import random
import tkinter

from enum import auto, StrEnum
from typing import Iterable, Sequence

import torch
import torch.nn
import torch.nn.functional





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
        return torch.tensor([dx, dy, danger_up, danger_down, danger_left, danger_right], dtype=torch.float32)

    def step(
        current_state: PlayingState,
        action: PlayingState.Action
    ) -> tuple[State, float]:
        updated_state = current_state.step(action)

        if isinstance(updated_state, LostState):
            return updated_state, -10.0

        if isinstance(updated_state, WonState):
            return updated_state, 100.0

        if current_state.apple_position != updated_state.apple_position:
            return updated_state, 5.0

        return updated_state, 1.0

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


def learn_reinforce(
    *,
    width: int = 5,
    height: int = 5,
    episodes: int = 10000,
    alpha: float = 1,
    gamma: float = 1
) -> None:
    policy = torch.nn.Sequential(
        torch.nn.Linear(6, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, len(PlayingState.Action)),
        torch.nn.Softmax(dim=-1)
    )

    progress = []

    for _ in range(episodes):
        initial_state = PlayingState.random_start(width=width, height=height)
        steps = evaluate_policy(initial_state, policy, gamma)

        for t, (_, _, _, returns, gradients) in enumerate(steps):
            for parameter, gradient in zip(policy.parameters(), gradients):
                parameter.data += alpha * (gamma ** t) * returns * gradient

        progress.append(steps[0][3])

    import matplotlib.pyplot as plt

    plt.plot(progress, ".")
    plt.show()

    root = tkinter.Tk()
    size = 20
    states = evaluate_policy(
        PlayingState.random_start(width=width, height=height),
        policy,
        gamma
    )

    canvas = tkinter.Canvas(
        root,
        width=width * size,
        height=height * size,
        bg='white'
    )

    canvas.pack()

    def animate(index=0):
        if index < len(states):
            draw_state(canvas, states[index][0], size)
            root.after(int(1 * 1000), lambda: animate(index + 1))

    animate()
    root.mainloop()

    return progress
