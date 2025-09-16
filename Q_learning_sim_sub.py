import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Enhanced preprocessing for multiple fraud types
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['CDR_START_DATE', 'SUBSCRIPTION_DATE', 'LAST_SIM_CHANGE_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Extract time-based features
    df['CALL_HOUR'] = df['CDR_START_DATE'].dt.hour
    df['CALL_DAY_OF_WEEK'] = df['CDR_START_DATE'].dt.dayofweek
    df['CALL_MONTH'] = df['CDR_START_DATE'].dt.month
    
    # Calculate days since subscription and last SIM change
    if 'SUBSCRIPTION_DATE' in df.columns:
        df['DAYS_SINCE_SUBSCRIPTION'] = (df['CDR_START_DATE'] - df['SUBSCRIPTION_DATE']).dt.days
    if 'LAST_SIM_CHANGE_DATE' in df.columns:
        df['DAYS_SINCE_SIM_CHANGE'] = (df['CDR_START_DATE'] - df['LAST_SIM_CHANGE_DATE']).dt.days
    
    # Normalize numerical features
    numerical_features = ['CDR_DURATION_SEC', 'CHARGE_FEE_1', 'DAYS_SINCE_SUBSCRIPTION', 
                          'DAYS_SINCE_SIM_CHANGE', 'CALL_HOUR', 'CALL_DAY_OF_WEEK', 'CALL_MONTH']
    
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    scaler = StandardScaler()
    for feature in numerical_features:
        if feature in df.columns:
            # Fill NaN values with mean before scaling
            df[feature].fillna(df[feature].mean(), inplace=True)
            df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    
    # Encode categorical features
    if 'OTHER_PARTY' in df.columns:
        df['OTHER_PARTY'] = df['OTHER_PARTY'].astype('category').cat.codes
    
    # Handle multiple fraud types
    if 'FRAUD_TYPE' in df.columns:
        # Convert fraud types to numerical codes
        fraud_mapping = {'Normal': 0, 'Subscription_Fraud': 1, 'SIM_Swap': 2}
        df['FRAUD_CODE'] = df['FRAUD_TYPE'].map(fraud_mapping)
        # Fill NaN with Normal (0)
        df['FRAUD_CODE'].fillna(0, inplace=True)
    elif 'FRAUD_STATUS' in df.columns:
        # If no fraud type column, assume binary fraud detection
        df['FRAUD_CODE'] = df['FRAUD_STATUS'].apply(lambda x: 0 if x == 'Normal' else 1)
    else:
        # If no fraud column at all, create a dummy one
        df['FRAUD_CODE'] = 0
    
    return df

# Enhanced Q-Learning Agent for multiple fraud types
class QLearningFraudDetection:
    def __init__(self, n_actions, n_states, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.n_actions = n_actions
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.state_mapping = defaultdict(int)
        self.state_counter = 0
    
    def get_state(self, row):
        # Create a state representation based on multiple features
        features = []
        
        # Add normalized features to state representation
        if 'CDR_DURATION_SEC' in row:
            features.append(int(row['CDR_DURATION_SEC'] * 10))
        if 'CALL_HOUR' in row:
            features.append(row['CALL_HOUR'])
        if 'OTHER_PARTY' in row:
            features.append(row['OTHER_PARTY'])
        if 'CHARGE_FEE_1' in row:
            features.append(int(row['CHARGE_FEE_1'] * 10))
        if 'CALL_DAY_OF_WEEK' in row:
            features.append(row['CALL_DAY_OF_WEEK'])
        if 'DAYS_SINCE_SUBSCRIPTION' in row:
            features.append(int(row['DAYS_SINCE_SUBSCRIPTION'] * 10))
        if 'DAYS_SINCE_SIM_CHANGE' in row:
            features.append(int(row['DAYS_SINCE_SIM_CHANGE'] * 10))
        
        # Create a unique state hash
        state_hash = hash(tuple(features)) % self.n_states
        
        # Track state mapping for analysis
        if state_hash not in self.state_mapping:
            self.state_mapping[state_hash] = self.state_counter
            self.state_counter += 1
            
        return state_hash

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def get_reward(self, row, action):
        true_label = row['FRAUD_CODE']
        
        # Different rewards for different fraud types
        if action == true_label:
            if true_label == 0:  # Correct normal classification
                return 5
            elif true_label == 1:  # Correct subscription fraud detection
                return 15
            elif true_label == 2:  # Correct SIM swap detection
                return 20
        else:
            if true_label == 0 and action != 0:  # False positive
                return -10
            elif true_label != 0 and action == 0:  # False negative (missed fraud)
                return -15
            else:  # Confusing two fraud types
                return -5

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_optimal_policy(self):
        return np.argmax(self.q_table, axis=1)

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    # Determine which classes are actually present
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Create appropriate labels and target names
    fraud_types = ['Normal', 'Subscription_Fraud', 'SIM_Swap']
    available_fraud_types = [fraud_types[i] for i in unique_classes if i < len(fraud_types)]
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=available_fraud_types, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=available_fraud_types, 
                yticklabels=available_fraud_types)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_excel("cdrs_t.xlsx")
        print("Columns in dataset:", df.columns.tolist())
    except FileNotFoundError:
        print("File not found. Please ensure 'cdrs_t.xlsx' exists in the current directory.")
        exit()
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Check what fraud types we have in the data
    print("Unique FRAUD_CODE values:", df['FRAUD_CODE'].unique())
    
    # Determine the number of actions based on available fraud types
    n_unique_fraud_types = len(df['FRAUD_CODE'].unique())
    n_actions = max(3, n_unique_fraud_types)  # At least 3, but more if needed
    
    # Calculate the number of unique states
    #unique_states = 1000  # Fixed size for state space, can be adjusted
    unique_states = df.apply(lambda row: (
       int(row['CDR_DURATION_SEC'] * 10),  # duration_state
       row['CALL_HOUR'],                   # hour_state
       row['OTHER_PARTY'],                 # other_party_state
       int(row['CHARGE_FEE_1'] * 10)       # charge_state
    ), axis=1).drop_duplicates().shape[0]
    
    print(f"Number of unique states: {unique_states}")
    print(f"Number of actions: {n_actions}")
    
    # Initialize Q-learning agent
    agent = QLearningFraudDetection(n_actions, unique_states)
    
    reward_log = []
    
    # Set epochs
    dataset_size = len(df)
    epochs = min(50, dataset_size)  # Use a reasonable number of epochs
    print(f"Number of epochs set to {epochs}")
    
    # Training loop
    for epoch in range(epochs):
        epoch_rewards = []
        # Shuffle the dataset for each epoch
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        for idx, row in shuffled_df.iterrows():
            state = agent.get_state(row)
            action = agent.choose_action(state)
            reward = agent.get_reward(row, action)
            next_state = agent.get_state(row)
            agent.update_q_table(state, action, reward, next_state)
            epoch_rewards.append(reward)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Log average reward for this epoch
        avg_reward = np.mean(epoch_rewards)
        reward_log.append(avg_reward)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Extract the optimal policy
    optimal_policy = agent.get_optimal_policy()
    
    # Apply the optimal policy to the dataset
    df['PREDICTED_ACTION'] = df.apply(lambda row: optimal_policy[agent.get_state(row)], axis=1)
    
    # Map actions to fraud types
    action_mapping = {0: 'Normal', 1: 'Subscription_Fraud', 2: 'SIM_Swap'}
    df['PREDICTED_FRAUD_TYPE'] = df['PREDICTED_ACTION'].apply(
        lambda x: action_mapping.get(x, f'Unknown_{x}')
    )
    
    # Evaluate model performance
    evaluate_model(df['FRAUD_CODE'], df['PREDICTED_ACTION'])
    
    # Visualizations
    plt.figure(figsize=(12, 8))
    plt.plot(reward_log)
    plt.title("Average Reward per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()
    
    # Q-table visualization (for first 20 states)
    plt.figure(figsize=(12, 8))
    sns.heatmap(agent.q_table[:20], annot=True, cmap='coolwarm', 
                xticklabels=[action_mapping.get(i, f'Action {i}') for i in range(n_actions)],
                yticklabels=range(20))
    plt.title("Q-table Heatmap (First 20 States)")
    plt.show()
    
    # Policy visualization
    plt.figure(figsize=(10, 6))
    policy_counts = pd.Series(optimal_policy).value_counts().sort_index()
    policy_counts.index = policy_counts.index.map(lambda x: action_mapping.get(x, f'Action {x}'))
    policy_counts.plot(kind='bar')
    plt.title("Optimal Policy Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.show()
    
    # Save the updated dataset with predictions
    output_columns = [col for col in df.columns if col not in ['FRAUD_CODE', 'PREDICTED_ACTION']]
    df[output_columns].to_excel("Telkom_Updated.xlsx", index=False)
    print("Updated dataset saved with predictions")
