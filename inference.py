import random
import numpy as np

class ReminderEnv:
    def __init__(self):
        self.steps = 0
        self.max_steps = 10
        self.reset()

    def reset(self):
        self.steps = 0

        # 0 = Executive, 1 = Procrastinator, 2 = Relaxed Student
        self.profile = random.choice([0, 1, 2])

        self.state = np.array([
            random.random(),  # busy
            random.random(),  # importance
            random.random(),  # deadline
            float(self.profile)
        ])

        return self.state

    def step(self, action):
        busy, importance, deadline, profile = self.state

        reward = 0.0

        # ACTION: 1 = REMIND, 0 = WAIT
        if action == 1:
            if profile == 0:  # Executive
                reward = 1.0 if busy < 0.7 else 0.0

            elif profile == 1:  # Procrastinator
                reward = 1.0 if importance > 0.7 else 0.3

            elif profile == 2:  # Relaxed Student
                reward = 1.0 if deadline < 0.5 else 0.5

        else:  # WAIT
            reward = 0.5 if deadline > 0.2 else 0.0

        # New state (same profile)
        self.state = np.array([
            random.random(),
            random.random(),
            random.random(),
            float(profile)
        ])

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, reward, done, False, {}

# ---------------- AI AGENT ----------------
def agent(state):
    busy, importance, deadline, profile = state

    # Procrastinator → remind if important
    if profile == 1 and importance > 0.6:
        return 1

    # Deadline near → remind
    elif deadline < 0.3:
        return 1

    # Relaxed student → remind only in evening (simulate with deadline)
    elif profile == 2 and deadline < 0.5:
        return 1

    else:
        return 0

# ---------------- RUN ----------------
if __name__ == "__main__":
    env = ReminderEnv()
    state = env.reset()

    total_reward = 0

    for step in range(10):
        action = agent(state)

        state, reward, done, _, _ = env.step(action)
        total_reward += reward

        action_name = "REMIND" if action == 1 else "WAIT"
        print(f"Step {step+1}: {action_name} → Reward: {reward}")

        if done:
            break

    score = total_reward / 10
    print("\nFinal Score:", round(score, 2))