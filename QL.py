#include action from this print(f"State {state}: {agent.q_table[state]}")
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the Excel file
file_path = "Telkom.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Ensure required columns exist
required_columns = ['CDR_DURATION_SEC', 'CDR_START_DATE', 'OTHER_PARTY', 'CHARGE_FEE_1', 'FRAUD_STATUS']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing in the dataset.")

# Preprocess data
def preprocess_data(df):
    # Convert CDR_START_DATE to datetime and extract hour of the day
    df['CALL_HOUR'] = pd.to_datetime(df['CDR_START_DATE']).dt.hour
    # Normalize numerical features (e.g., call duration, frequency, charge rate)
    df['CDR_DURATION_SEC'] = (df['CDR_DURATION_SEC'] - df['CDR_DURATION_SEC'].min()) / (df['CDR_DURATION_SEC'].max() - df['CDR_DURATION_SEC'].min())
    df['CHARGE_FEE_1'] = (df['CHARGE_FEE_1'] - df['CHARGE_FEE_1'].min()) / (df['CHARGE_FEE_1'].max() - df['CHARGE_FEE_1'].min())
    # Encode categorical features (e.g., OTHER_PARTY)
    df['OTHER_PARTY'] = df['OTHER_PARTY'].astype('category').cat.codes

    return df

df = preprocess_data(df)

# Q-Learning setup
class QLearningFraudDetection:
    def __init__(self, n_actions, n_states, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))  # Q-table initialization

    def get_state(self, row):
        # Define state based on multiple features
        duration_state = int(row['CDR_DURATION_SEC'] * 10)  # Scale normalized duration to 0-10
        #frequency_state = int(row['CALL_FREQUENCY'] * 10)  # Scale normalized frequency to 0-10
        hour_state = row['CALL_HOUR']  # Hour of the day (0-23)
        other_party_state = row['OTHER_PARTY']  # Encoded OTHER_PARTY
        charge_state = int(row['CHARGE_FEE_1'] * 10)  # Scale normalized charge rate to 0-10
        #df['optimal_policy'] = np.argmax(self.q_table, axis=1)
        # Combine features into a single state (you can adjust the weights or logic)
        state = duration_state + hour_state + other_party_state + charge_state
        return state % self.n_states  # Ensure state is within n_states

    def choose_action(self, state):
        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Random action (exploration)
        else:
            return np.argmax(self.q_table[state])  # Best-known action (exploitation)

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

    def get_reward(self, row, action):
        # Reward based on the action taken and ground truth
        if action == 1 and row['FRAUD_STATUS'] == 'Fraud':
            return 10  # Positive reward for correctly detecting fraud
        elif action == 0 and row['FRAUD_STATUS'] == 'Normal':
            return 10  # Positive reward for correctly classifying as normal
        else:
            return -10  # Negative reward for incorrect classification

# Initialize Q-learning agent
n_actions = 2  # Fraud (1) or Normal (0)
n_states = 599  # Increased number of states to accommodate more features
agent = QLearningFraudDetection(n_actions, n_states)

# Initialize 'FRAUD_STATUS' and 'COMMENT' columns if they don't exist
if 'FRAUD_STATUS' not in df.columns or 'COMMENT' not in df.columns:
    # Initialize columns with default values
    df['FRAUD_STATUS'] = 'Normal'
    df['COMMENT'] = 'Normal'

    # Use get_state to determine initial fraud risk and update COMMENT
    for idx, row in df.iterrows():
        state = agent.get_state(row)
        if state == 2:  # High risk state (e.g., high duration, high charge rate, etc.)
            df.at[idx, 'FRAUD_STATUS'] = 'Fraud'
            df.at[idx, 'COMMENT'] = f"High risk detected based on initial state: {state}"
        elif state == 1:  # Medium risk state
            df.at[idx, 'FRAUD_STATUS'] = 'Suspicious'
            df.at[idx, 'COMMENT'] = f"Medium risk detected based on initial state: {state}"
        else:  # Low risk state
            df.at[idx, 'FRAUD_STATUS'] = 'Normal'
            df.at[idx, 'COMMENT'] = f"Low risk detected based on initial state: {state}"

# Precompute states for each row
df['state'] = df.apply(agent.get_state, axis=1)

# Define the policy
def apply_policy(row, agent):
    state = row['state']
    action = agent.choose_action(state)

    # Apply the action
    if action == 1:  # Fraud action
        row['FRAUD_STATUS'] = 'Fraud'
        row['COMMENT'] = f"Fraud detected based on features: Duration={row['CDR_DURATION_SEC']}, Hour={row['CALL_HOUR']}, OTHER_PARTY={row['OTHER_PARTY']}, Charge={row['CHARGE_FEE_1']}"
    else:  # Normal action
        row['FRAUD_STATUS'] = 'Normal'
        row['COMMENT'] = 'Low risk'

    return row

# Initialize a list to track rewards for analysis
reward_log = []

# Training loop
epochs = 599  # Increased number of epochs for better training
for epoch in range(epochs):
    for idx, row in df.iterrows():
        state = row['state']
        action = agent.choose_action(state)

        # Apply the action based on the chosen policy
        df.loc[idx] = apply_policy(row, agent)

        # Get reward and next state
        reward = agent.get_reward(row, action)
        reward_log.append(reward)  # Log reward

        next_state = row['state']
        agent.update_q_table(state, action, reward, next_state)
        
# Display Q-table and reward log after training
print("\nFinal Q-table:")
for state in range(agent.n_states):
    #print(f"State {state}: {agent.q_table[state]}")
    print(f"State {state}: {agent.q_table[state]}, Chosen Action: {action}")
#print(agent.q_table)
#print("Optimal Policy (Action for Each State):", optimal_policy)

# Plot Q-table heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(agent.q_table, annot=True, cmap='coolwarm', xticklabels=["Normal", "Fraud"], yticklabels=range(n_states))
plt.title("Q-table Heatmap Visualisation")
plt.xlabel("Actions")
plt.ylabel("States")
plt.show()

# Plot reward distribution (histogram)
plt.figure(figsize=(8, 6))
plt.hist(reward_log, bins=20, edgecolor='black', alpha=0.7)
plt.title("Reward Algorithm ")
plt.xlabel("Reward Value")
plt.ylabel("Frequency")
plt.show()

# Save the updated Excel file with predictions
df.to_excel(file_path, index=False)

print(f"Updated Excel file saved at {file_path}")
print(df.head())