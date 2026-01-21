import gymnasium
import torch


def iterate_policy(policy, environment):
    current_observation, _ = environment.reset()

    while True:
        y = policy(torch.Tensor(current_observation))
        action = torch.multinomial(y, 1).item()
        z = torch.log(y[action])
        z.backward()
        gradients = [
            parameter.grad.clone()
            for parameter in policy.parameters()
        ]

        policy.zero_grad()

        updated_observation, reward, terminated, truncated, _ = \
            environment.step(action)

        yield current_observation, action, reward, gradients

        if terminated or truncated:
            return

        current_observation = updated_observation


def evaluate_policy(policy, environment, gamma):
    steps = list(iterate_policy(policy, environment))
    result = []

    for i, (observation, action, reward, gradients) in enumerate(steps):
        returns = sum(
            (gamma ** j) * reward
            for j, (_, _, reward, _) in enumerate(steps[i:])
        )

        item = (observation, action, reward, returns, gradients)
        result.append(item)

    return result


def learn_reinforce_with_baseline(
    *,
    episodes: int = 5000,
    alpha_policy: float = 1e-5,
    alpha_evaluator: float = 1e-5,
    gamma: float = 1
) -> None:
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.Linear(64, 2),
        torch.nn.Softmax(dim=-1)
    )

    evaluator = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.Linear(64, 1)
    )

    environment = gymnasium.make("CartPole-v1")

    progress = []

    for _ in range(episodes):
        print(_)
        steps = evaluate_policy(policy, environment, gamma)

        for t, (observation, _, _, returns, gradients) in enumerate(steps):
            value = evaluator(torch.Tensor(observation))
            delta = returns - value.item()

            value.backward()

            for parameter in evaluator.parameters():
                parameter.data += alpha_evaluator * delta * parameter.grad

            evaluator.zero_grad()

            for parameter, gradient in zip(policy.parameters(), gradients):
                parameter.data += \
                    alpha_policy * (gamma ** t) * delta * gradient

        progress.append(steps[0][3])


    import matplotlib.pyplot as plt

    plt.plot(progress)
    plt.show()

    environment = gymnasium.make("CartPole-v1", render_mode="human")

    evaluate_policy(policy, environment, gamma)


learn_reinforce_with_baseline()
