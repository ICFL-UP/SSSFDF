import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Enhanced preprocessing for unsupervised learning
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['CDR_START_DATE', 'SUBSCRIPTION_DATE', 'LAST_SIM_CHANGE_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Extract time-based features
    if 'CDR_START_DATE' in df.columns:
        df['CALL_HOUR'] = df['CDR_START_DATE'].dt.hour
        df['CALL_DAY_OF_WEEK'] = df['CDR_START_DATE'].dt.dayofweek
        df['CALL_MONTH'] = df['CDR_START_DATE'].dt.month
    
    # Calculate days since subscription and last SIM change
    if 'SUBSCRIPTION_DATE' in df.columns and 'CDR_START_DATE' in df.columns:
        df['DAYS_SINCE_SUBSCRIPTION'] = (df['CDR_START_DATE'] - df['SUBSCRIPTION_DATE']).dt.days
    if 'LAST_SIM_CHANGE_DATE' in df.columns and 'CDR_START_DATE' in df.columns:
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
    
    # Remove any fraud-related columns since we're doing unsupervised learning
    fraud_columns = ['FRAUD_TYPE', 'FRAUD_STATUS', 'FRAUD_CODE']
    for col in fraud_columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return df

# Unsupervised Q-Learning Agent for anomaly detection
class UnsupervisedQLearning:
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
        self.anomaly_scores = {}
        self.cluster_model = None
    
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

    def calculate_anomaly_score(self, row):
        """Calculate anomaly score based on feature deviations"""
        score = 0
        
        # Duration anomaly (very short or very long calls)
        if 'CDR_DURATION_SEC' in row:
            duration = row['CDR_DURATION_SEC']
            if duration > 2 or duration < -2:  # More than 2 std deviations
                score += abs(duration)
        
        # Unusual call hours (outside normal business hours)
        if 'CALL_HOUR' in row:
            hour = row['CALL_HOUR']
            if hour < 6 or hour > 22:  # Calls between 10 PM and 6 AM
                score += 2
        
        # High charge anomalies
        if 'CHARGE_FEE_1' in row:
            charge = row['CHARGE_FEE_1']
            if charge > 2:  # More than 2 std deviations
                score += abs(charge)
        
        # Recent subscription or SIM change
        if 'DAYS_SINCE_SUBSCRIPTION' in row:
            days_sub = row['DAYS_SINCE_SUBSCRIPTION']
            if days_sub < -1:  # Very recent subscription
                score += 2
        
        if 'DAYS_SINCE_SIM_CHANGE' in row:
            days_sim = row['DAYS_SINCE_SIM_CHANGE']
            if days_sim < -1:  # Very recent SIM change
                score += 2
        
        return score

    def get_reward(self, row, action, state_visit_count):
        """Unsupervised reward function based on anomaly detection"""
        anomaly_score = self.calculate_anomaly_score(row)
        state = self.get_state(row)
        
        # Reward based on anomaly score and state frequency
        if action == 0:  # Classify as normal
            if anomaly_score < 3:  # Low anomaly score
                return 5  # Reward for correct normal classification
            else:
                return -10  # Penalty for missing anomaly
        
        elif action == 1:  # Classify as suspicious
            if anomaly_score >= 3 and anomaly_score < 6:  # Medium anomaly
                return 10
            elif anomaly_score < 3:  # Low anomaly (false positive)
                return -5
            else:  # High anomaly (should be fraud)
                return 5
        
        else:  # Classify as fraud (action >= 2)
            if anomaly_score >= 6:  # High anomaly score
                return 15  # High reward for detecting clear anomalies
            elif anomaly_score >= 3:  # Medium anomaly
                return 8
            else:  # Low anomaly (false positive)
                return -8
        
        # Additional reward for exploring rare states
        if state_visit_count.get(state, 0) < 5:  # Rare state
            return reward + 2
        
        return reward

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_optimal_policy(self):
        return np.argmax(self.q_table, axis=1)

# Function to evaluate unsupervised model performance
def evaluate_unsupervised_model(df, predicted_actions):
    # Create anomaly score for evaluation
    df['ANOMALY_SCORE'] = df.apply(lambda row: 
        abs(row['CDR_DURATION_SEC']) + 
        abs(row['CHARGE_FEE_1']) + 
        (1 if row.get('CALL_HOUR', 12) < 6 or row.get('CALL_HOUR', 12) > 22 else 0), 
        axis=1)
    
    # Classify based on anomaly score for comparison
    df['ANOMALY_BASED_PREDICTION'] = pd.cut(df['ANOMALY_SCORE'], 
                                           bins=[-1, 2, 5, 100], 
                                           labels=[0, 1, 2])
    
    print("Anomaly Score Distribution:")
    print(df['ANOMALY_SCORE'].describe())
    
    # Compare Q-learning predictions with anomaly-based predictions
    agreement = (predicted_actions == df['ANOMALY_BASED_PREDICTION']).mean()
    print(f"Agreement with anomaly-based method: {agreement:.2%}")
    
    # Visualize predictions vs anomaly scores
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for action in range(3):
        action_data = df[predicted_actions == action]
        plt.scatter(action_data.index, action_data['ANOMALY_SCORE'], 
                   label=f'Action {action}', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.title('Q-learning Predictions vs Anomaly Score')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    df['ANOMALY_SCORE'].hist(bins=50, alpha=0.7)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution')
    
    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_excel("test_data.xlsx")
        print("Columns in dataset:", df.columns.tolist())
    except FileNotFoundError:
        print("File not found. Please ensure 'test_data.xlsx' exists in the current directory.")
        exit()
    
    # Preprocess data (removes fraud labels)
    df = preprocess_data(df)
    print("Data preprocessing completed. Fraud labels removed for unsupervised learning.")
    
    # Calculate the number of unique states
    unique_states = df.apply(lambda row: (
       int(row.get('CDR_DURATION_SEC', 0) * 10),
       row.get('CALL_HOUR', 12),
       row.get('OTHER_PARTY', 0),
       int(row.get('CHARGE_FEE_1', 0) * 10)
    ), axis=1).drop_duplicates().shape[0]
    
    n_actions = 3  # 0: Normal, 1: Suspicious, 2: Fraud
    
    print(f"Number of unique states: {unique_states}")
    print(f"Number of actions: {n_actions}")
    
    # Initialize unsupervised Q-learning agent
    agent = UnsupervisedQLearning(n_actions, unique_states)
    
    reward_log = []
    state_visit_count = defaultdict(int)
    
    # Set epochs
    dataset_size = len(df)
    epochs = min(10, dataset_size)
    print(f"Number of epochs set to {epochs}")
    
    # Training loop
    for epoch in range(epochs):
        epoch_rewards = []
        # Shuffle the dataset for each epoch
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        for idx, row in shuffled_df.iterrows():
            state = agent.get_state(row)
            state_visit_count[state] += 1
            action = agent.choose_action(state)
            reward = agent.get_reward(row, action, state_visit_count)
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
    action_mapping = {0: 'Normal', 1: 'Suspicious', 2: 'Fraud'}
    df['PREDICTED_FRAUD_TYPE'] = df['PREDICTED_ACTION'].map(action_mapping)
    
    # Evaluate unsupervised model performance
    evaluate_unsupervised_model(df, df['PREDICTED_ACTION'])
    
    # Print prediction distribution
    print("\nPrediction Distribution:")
    print(df['PREDICTED_FRAUD_TYPE'].value_counts())
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Reward progression
    plt.subplot(2, 2, 1)
    plt.plot(reward_log)
    plt.title("Average Reward per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.grid(True)
    
    # Q-table visualization (for first 20 states)
    plt.subplot(2, 2, 2)
    sns.heatmap(agent.q_table[:20], annot=True, cmap='coolwarm', 
                xticklabels=[action_mapping[i] for i in range(n_actions)],
                yticklabels=range(20))
    plt.title("Q-table Heatmap (First 20 States)")
    
    # Policy visualization
    plt.subplot(2, 2, 3)
    policy_counts = pd.Series(optimal_policy).value_counts().sort_index()
    policy_counts.index = policy_counts.index.map(action_mapping)
    policy_counts.plot(kind='bar')
    plt.title("Optimal Policy Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    
    # Anomaly score distribution by predicted class
    plt.subplot(2, 2, 4)
    df['ANOMALY_SCORE'] = df.apply(agent.calculate_anomaly_score, axis=1)
    for action_type in action_mapping.values():
        action_data = df[df['PREDICTED_FRAUD_TYPE'] == action_type]
        plt.hist(action_data['ANOMALY_SCORE'], alpha=0.7, label=action_type, bins=20)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score by Predicted Class')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save the updated dataset with predictions
    output_columns = [col for col in df.columns if col not in ['PREDICTED_ACTION']]
    df[output_columns].to_excel("Telkom_Unsupervised_Predictions.xlsx", index=False)
    print("Updated dataset saved with unsupervised predictions")