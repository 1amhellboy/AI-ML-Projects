import gymnasium as gym

# Create CartPole environment
env = gym.make("CartPole-v1", render_mode="ansi")

# Reset environment → get initial state
state, _ = env.reset()
print("Initial State:", state)

# Example: take a random action
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)

print("Next State:", next_state)
print("Reward:", reward)
print("Done:", terminated or truncated)
