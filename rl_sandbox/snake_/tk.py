import tkinter
import queue

from .snake import PlayingState


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
        print(self.state)

        if not isinstance(self.state, PlayingState):
            return

        try:
            action = self.actions.get(False)
        except queue.Empty:
            action = PlayingState.Action.IDLE

        self.state = self.state.step(action)

        self.root.after(300, self.update)

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


def main():
    root = tkinter.Tk()
    app = App(root, 10, 10, 30)

    app.start()
    root.mainloop()


if __name__ == "__main__":
    main()
