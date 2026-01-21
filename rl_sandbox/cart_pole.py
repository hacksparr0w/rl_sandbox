import gymnasium
import torch
import matplotlib.pyplot as plt


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


def learn_actor_critic(
    *,
    episodes: int = 2000,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    gamma: float = 0.99
) -> None:
    actor = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
        torch.nn.Softmax(dim=-1)
    )

    critic = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )

    environment = gymnasium.make("CartPole-v1")
    progress = []

    for episode in range(episodes):
        current_observation, _ = environment.reset()
        current_observation = torch.Tensor(current_observation)
        total_reward = 0

        while True:
            p = actor(current_observation)
            action = torch.multinomial(p, 1).item()
            z = torch.log(p[action])
            z.backward()
            gradients = [
                parameter.grad.clone()
                for parameter in actor.parameters()
            ]

            actor.zero_grad()

            updated_observation, reward, terminated, truncated, _ = \
                environment.step(action)

            total_reward += reward
            updated_observation = torch.Tensor(updated_observation)

            current_value = critic(current_observation)
            updated_value = critic(updated_observation)

            # Bootstrap is 0 at terminal states
            next_val = 0 if terminated else updated_value.item()
            delta = reward + gamma * next_val - current_value.item()

            for parameter, gradient in zip(actor.parameters(), gradients):
                parameter.data += \
                    actor_lr * delta * gradient

            current_value.backward()

            for parameter in critic.parameters():
                parameter.data += critic_lr * delta * parameter.grad

            critic.zero_grad()

            if terminated or truncated:
                break

            current_observation = updated_observation

        progress.append(total_reward)
        print(f"Episode {episode} finished with total reward of: {total_reward}")
    
    plt.plot(progress)
    plt.show()

    # Render the trained agent
    env_render = gymnasium.make("CartPole-v1", render_mode="human")
    obs, _ = env_render.reset()
    obs = torch.Tensor(obs)

    while True:
        with torch.no_grad():
            p = actor(obs)
            action = torch.argmax(p).item()

        obs, _, terminated, truncated, _ = env_render.step(action)
        obs = torch.Tensor(obs)

        if terminated or truncated:
            break

    env_render.close()


learn_reinforce_with_baseline()
