import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time
import warnings
import xgboost as xgb
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

# Enhanced preprocessing for unsupervised learning with better feature engineering
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
        df['IS_WEEKEND'] = df['CDR_START_DATE'].dt.dayofweek >= 5
        df['IS_NIGHT'] = (df['CDR_START_DATE'].dt.hour < 6) | (df['CDR_START_DATE'].dt.hour > 22)
    
    # Calculate days since subscription and last SIM change
    if 'SUBSCRIPTION_DATE' in df.columns and 'CDR_START_DATE' in df.columns:
        df['DAYS_SINCE_SUBSCRIPTION'] = (df['CDR_START_DATE'] - df['SUBSCRIPTION_DATE']).dt.days
        df['IS_NEW_SUBSCRIBER'] = df['DAYS_SINCE_SUBSCRIPTION'] < 30
    if 'LAST_SIM_CHANGE_DATE' in df.columns and 'CDR_START_DATE' in df.columns:
        df['DAYS_SINCE_SIM_CHANGE'] = (df['CDR_START_DATE'] - df['LAST_SIM_CHANGE_DATE']).dt.days
        df['RECENT_SIM_CHANGE'] = df['DAYS_SINCE_SIM_CHANGE'] < 7
    
    # Create interaction features that might indicate fraud patterns
    if all(col in df.columns for col in ['CDR_DURATION_SEC', 'CHARGE_FEE_1']):
        df['DURATION_TO_CHARGE_RATIO'] = df['CDR_DURATION_SEC'] / (df['CHARGE_FEE_1'] + 0.001)
        df['HIGH_COST_SHORT_CALL'] = (df['CDR_DURATION_SEC'] < 30) & (df['CHARGE_FEE_1'] > df['CHARGE_FEE_1'].quantile(0.75))
    
    # Normalize numerical features
    numerical_features = ['CDR_DURATION_SEC', 'CHARGE_FEE_1', 'DAYS_SINCE_SUBSCRIPTION', 
                          'DAYS_SINCE_SIM_CHANGE', 'CALL_HOUR', 'CALL_DAY_OF_WEEK', 'CALL_MONTH',
                          'DURATION_TO_CHARGE_RATIO']
    
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    scaler = StandardScaler()
    for feature in numerical_features:
        if feature in df.columns:
            # Fill NaN values with mean before scaling
            df[feature].fillna(df[feature].mean(), inplace=True)
            df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    
    # Encode categorical features with frequency encoding for rare categories
    if 'OTHER_PARTY' in df.columns:
        # Frequency encoding for OTHER_PARTY
        party_counts = df['OTHER_PARTY'].value_counts()
        df['OTHER_PARTY_FREQ'] = df['OTHER_PARTY'].map(party_counts)
        df['OTHER_PARTY_RARE'] = df['OTHER_PARTY_FREQ'] < 5  # Flag rare parties
    
    # Remove any fraud-related columns since we're doing unsupervised learning
    fraud_columns = ['FRAUD_TYPE', 'FRAUD_STATUS', 'FRAUD_CODE']
    for col in fraud_columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return df

# Enhanced Unsupervised Q-Learning Agent with class imbalance handling
class UnsupervisedQLearning:
    def __init__(self, n_actions, n_states, learning_rate=0.1, discount_factor=0.9, epsilon=0.3, epsilon_decay=0.998, min_epsilon=0.05):
        self.n_actions = n_actions
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Higher initial epsilon for more exploration
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.state_mapping = defaultdict(int)
        self.state_counter = 0
        self.anomaly_scores = {}
        self.cluster_model = None
        # Class weights to handle imbalance: Higher rewards for correct minority class predictions
        self.class_weights = {0: 1.0, 1: 3.0, 2: 5.0}  # Normal, Suspicious, Fraud
    
    def get_state(self, row):
        # Enhanced state representation with more features
        features = []
        
        # Add normalized features to state representation
        if 'CDR_DURATION_SEC' in row:
            features.append(int(row['CDR_DURATION_SEC'] * 10))
        if 'CALL_HOUR' in row:
            features.append(row['CALL_HOUR'])
        if 'OTHER_PARTY' in row:
            features.append(row['OTHER_PARTY'] % 100)  # Reduce cardinality
        if 'CHARGE_FEE_1' in row:
            features.append(int(row['CHARGE_FEE_1'] * 10))
        if 'CALL_DAY_OF_WEEK' in row:
            features.append(row['CALL_DAY_OF_WEEK'])
        if 'DAYS_SINCE_SUBSCRIPTION' in row:
            features.append(int(row['DAYS_SINCE_SUBSCRIPTION'] * 10))
        if 'DAYS_SINCE_SIM_CHANGE' in row:
            features.append(int(row['DAYS_SINCE_SIM_CHANGE'] * 10))
        if 'IS_NIGHT' in row:
            features.append(int(row['IS_NIGHT']))
        if 'RECENT_SIM_CHANGE' in row:
            features.append(int(row['RECENT_SIM_CHANGE']))
        
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
        """Enhanced anomaly score calculation with more fraud indicators"""
        score = 0
        
        # Duration anomaly (very short or very long calls)
        if 'CDR_DURATION_SEC' in row:
            duration = row['CDR_DURATION_SEC']
            if duration > 2.5 or duration < -2.5:  # More than 2.5 std deviations
                score += abs(duration) * 1.5
        
        # Unusual call hours (outside normal business hours)
        if 'CALL_HOUR' in row:
            hour = row['CALL_HOUR']
            if hour < 5 or hour > 23:  # Calls between 11 PM and 5 AM
                score += 2.5
        
        # High charge anomalies
        if 'CHARGE_FEE_1' in row:
            charge = row['CHARGE_FEE_1']
            if charge > 2.5:  # More than 2.5 std deviations
                score += abs(charge) * 2.0
        
        # Recent subscription or SIM change (strong fraud indicators)
        if 'DAYS_SINCE_SUBSCRIPTION' in row:
            days_sub = row['DAYS_SINCE_SUBSCRIPTION']
            if days_sub < -1.5:  # Very recent subscription
                score += 3.0
        
        if 'DAYS_SINCE_SIM_CHANGE' in row:
            days_sim = row['DAYS_SINCE_SIM_CHANGE']
            if days_sim < -1.5:  # Very recent SIM change
                score += 3.0
        
        # Night calls
        if 'IS_NIGHT' in row and row['IS_NIGHT']:
            score += 1.5
            
        # Weekend calls pattern
        if 'IS_WEEKEND' in row and row['IS_WEEKEND']:
            score += 0.5
            
        # Rare other party
        if 'OTHER_PARTY_RARE' in row and row['OTHER_PARTY_RARE']:
            score += 2.0
            
        # High cost short calls
        if 'HIGH_COST_SHORT_CALL' in row and row['HIGH_COST_SHORT_CALL']:
            score += 3.0
        
        return score

    def get_reward(self, row, action, state_visit_count):
        """Enhanced reward function with class weighting for imbalance handling"""
        anomaly_score = self.calculate_anomaly_score(row)
        state = self.get_state(row)
        
        base_reward = 0
        
        # Enhanced reward structure with class weights
        if action == 0:  # Classify as normal
            if anomaly_score < 2.5:  # Low anomaly score
                base_reward = 3 * self.class_weights[0]
            else:
                base_reward = -8 * self.class_weights[0]  # Higher penalty for missing anomalies
        
        elif action == 1:  # Classify as suspicious
            if anomaly_score >= 2.5 and anomaly_score < 5.5:  # Medium anomaly
                base_reward = 8 * self.class_weights[1]
            elif anomaly_score < 2.5:  # Low anomaly (false positive)
                base_reward = -4 * self.class_weights[1]
            else:  # High anomaly (should be fraud)
                base_reward = 4 * self.class_weights[1]
        
        else:  # Classify as fraud (action >= 2)
            if anomaly_score >= 5.5:  # High anomaly score
                base_reward = 12 * self.class_weights[2]  # High reward for detecting clear anomalies
            elif anomaly_score >= 2.5:  # Medium anomaly
                base_reward = 6 * self.class_weights[2]
            else:  # Low anomaly (false positive)
                base_reward = -6 * self.class_weights[2]
        
        # Additional reward for exploring rare states (encourage exploration of fraud patterns)
        state_freq = state_visit_count.get(state, 0)
        if state_freq < 3:  # Rare state - likely unusual patterns
            exploration_bonus = 3
        elif state_freq > 20:  # Very common state - likely normal patterns
            exploration_bonus = -1  # Small penalty for over-exploring common states
        else:
            exploration_bonus = 0
            
        return base_reward + exploration_bonus

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_optimal_policy(self):
        return np.argmax(self.q_table, axis=1)
    
    def get_action_probabilities(self, state):
        # Convert Q-values to probabilities using softmax with temperature
        temperature = 0.7  # Lower temperature for more confident predictions
        q_values = self.q_table[state] / temperature
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        return exp_q / np.sum(exp_q)

# XGBoost Model with advanced imbalance handling - CORRECTED VERSION
class XGBoostFraudDetector:
    def __init__(self, use_smote=True):
        self.use_smote = use_smote
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def prepare_features(self, df):
        """Prepare features for XGBoost training"""
        # Select relevant features
        feature_columns = [
            'CDR_DURATION_SEC', 'CHARGE_FEE_1', 'CALL_HOUR', 'CALL_DAY_OF_WEEK',
            'CALL_MONTH', 'DAYS_SINCE_SUBSCRIPTION', 'DAYS_SINCE_SIM_CHANGE',
            'IS_WEEKEND', 'IS_NIGHT', 'IS_NEW_SUBSCRIBER', 'RECENT_SIM_CHANGE',
            'DURATION_TO_CHARGE_RATIO', 'HIGH_COST_SHORT_CALL', 'OTHER_PARTY_FREQ',
            'OTHER_PARTY_RARE'
        ]
        
        # Only use columns that exist in the dataframe
        available_features = [f for f in feature_columns if f in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].copy()
        
        # Fill any remaining NaN values
        X = X.fillna(X.mean())
        
        return X
    
    def handle_imbalance(self, X, y):
        """Apply SMOTE or other techniques to handle class imbalance"""
        if not self.use_smote:
            return X, y
            
        try:
            # Check if we have at least 2 samples for each class
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples_needed = 6  # SMOTE requires at least 6 samples per class
            
            if all(count >= min_samples_needed for count in class_counts):
                smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts)-1))
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"SMOTE applied: Original shape {X.shape}, Resampled shape {X_resampled.shape}")
                return X_resampled, y_resampled
            else:
                print("Insufficient samples for SMOTE. Using original data.")
                return X, y
        except Exception as e:
            print(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalance handling"""
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        class_weights = {}
        
        for i, class_label in enumerate(unique):
            class_weights[class_label] = total_samples / (len(unique) * counts[i])
        
        print(f"Class distribution: {dict(zip(unique, counts))}")
        print(f"Calculated class weights: {class_weights}")
        
        return class_weights
    
    def train(self, df, y):
        """Train XGBoost model with imbalance handling"""
        X = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle imbalance
        X_train_resampled, y_train_resampled = self.handle_imbalance(X_train, y_train)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train_resampled)
        
        # XGBoost parameters for imbalanced data - CORRECTED
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        # Train model - CORRECTED: early_stopping_rounds is in fit() method in newer versions
        self.model = xgb.XGBClassifier(**xgb_params)
        
        start_time = time.time()
        
        # Use the correct fit method without early_stopping_rounds parameter
        self.model.fit(X_train_resampled, y_train_resampled)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        print("XGBoost Test Set Performance:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Subscription_Fraud', 'SIM_Swap_Fraud']))
        
        return training_time, X_test, y_test, y_pred, y_pred_proba
    
    def predict(self, df):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities

# Enhanced evaluation function for both models
def evaluate_model_comprehensive(df, predicted_actions, probabilities=None, model_name="Model", 
                               fit_time=None, predict_time=None, y_true_proxy=None):
    
    if y_true_proxy is None:
        # Create enhanced anomaly score for evaluation
        df['ANOMALY_SCORE'] = df.apply(lambda row: 
            abs(row.get('CDR_DURATION_SEC', 0)) * 1.5 + 
            abs(row.get('CHARGE_FEE_1', 0)) * 2.0 + 
            (2.5 if row.get('IS_NIGHT', False) else 0) +
            (3.0 if row.get('RECENT_SIM_CHANGE', False) else 0) +
            (2.0 if row.get('OTHER_PARTY_RARE', False) else 0), 
            axis=1)
        
        # Adjust classification thresholds based on anomaly score distribution
        score_75 = df['ANOMALY_SCORE'].quantile(0.75)
        score_90 = df['ANOMALY_SCORE'].quantile(0.90)
        
        # Use dynamic thresholds based on data distribution
        df['ANOMALY_BASED_PREDICTION'] = pd.cut(df['ANOMALY_SCORE'], 
                                               bins=[-1, score_75, score_90, float('inf')], 
                                               labels=[0, 1, 2])
        
        print("Anomaly Score Distribution:")
        print(df['ANOMALY_SCORE'].describe())
        print(f"Dynamic Thresholds - Subscription: >{score_75:.2f}, SIM_Swap: >{score_90:.2f}")
        
        y_true_proxy = df['ANOMALY_BASED_PREDICTION'].values
    
    y_pred = predicted_actions
    
    # Classification Report
    print("\n" + "="*60)
    print(f"COMPREHENSIVE EVALUATION - {model_name}")
    print("="*60)
    target_names = ['Normal', 'Subscription_Fraud', 'SIM_Swap_Fraud']
    print(classification_report(y_true_proxy, y_pred, target_names=target_names, zero_division=0))
    
    # Sensitivity (Recall) per class
    cm = confusion_matrix(y_true_proxy, y_pred, labels=[0, 1, 2])
    sensitivity_per_class = {}
    for i, class_idx in enumerate([0, 1, 2]):
        if np.sum(cm[i, :]) == 0:
            sensitivity_per_class[class_idx] = 0
        else:
            sensitivity_per_class[class_idx] = cm[i, i] / np.sum(cm[i, :])
    
    print("Sensitivity (Recall) per class:")
    for class_idx, sens in sensitivity_per_class.items():
        class_name = target_names[class_idx]
        print(f"  {class_name}: {sens:.4f}")
    
    # Calculate overall sensitivity (macro-average)
    overall_sensitivity = np.mean(list(sensitivity_per_class.values()))
    print(f"Overall Sensitivity (Macro-average): {overall_sensitivity:.4f}")
    
    # ROC-AUC and PR-AUC if probabilities are available
    if probabilities is not None:
        # For multi-class ROC-AUC and PR-AUC
        y_true_bin = np.eye(3)[y_true_proxy.astype(int)]  # One-hot encoding
        
        # Calculate ROC-AUC for each class
        roc_auc_per_class = {}
        for i in range(3):
            if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists in true labels
                roc_auc_per_class[i] = roc_auc_score(y_true_bin[:, i], probabilities[:, i])
            else:
                roc_auc_per_class[i] = 0.5  # Default for single class
        
        # Calculate macro-average ROC-AUC
        macro_roc_auc = roc_auc_score(y_true_bin, probabilities, average='macro', multi_class='ovr')
        
        print("\nROC-AUC Scores:")
        for class_idx, auc_score in roc_auc_per_class.items():
            class_name = target_names[class_idx]
            print(f"  {class_name}: {auc_score:.4f}")
        print(f"Macro-average ROC-AUC: {macro_roc_auc:.4f}")
        
        # Calculate PR-AUC for each class
        pr_auc_per_class = {}
        for i in range(3):
            if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists in true labels
                pr_auc_per_class[i] = average_precision_score(y_true_bin[:, i], probabilities[:, i])
            else:
                pr_auc_per_class[i] = 0  # Default for single class
        
        # Calculate macro-average PR-AUC
        macro_pr_auc = average_precision_score(y_true_bin, probabilities, average='macro')
        
        print("\nPR-AUC Scores:")
        for class_idx, auc_score in pr_auc_per_class.items():
            class_name = target_names[class_idx]
            print(f"  {class_name}: {auc_score:.4f}")
        print(f"Macro-average PR-AUC: {macro_pr_auc:.4f}")
    
    # Display timing information
    if fit_time is not None:
        print(f"\nTraining Time: {fit_time:.4f} seconds")
    if predict_time is not None:
        print(f"Prediction Time: {predict_time:.4f} seconds")
        if len(y_true_proxy) > 0:
            print(f"Latency per sample: {predict_time/len(y_true_proxy)*1000:.4f} milliseconds")
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return {
        'sensitivity': sensitivity_per_class,
        'overall_sensitivity': overall_sensitivity,
        'roc_auc': roc_auc_per_class if probabilities is not None else None,
        'pr_auc': pr_auc_per_class if probabilities is not None else None
    }

# Main Execution
if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_excel("cdrs_with_subscription_and_sim_change.xlsx")
        print("Columns in dataset:", df.columns.tolist())
    except FileNotFoundError:
        print("File not found. Please ensure 'cdrs_with_subscription_and_sim_change.xlsx' exists in the current directory.")
        exit()
    
    # Enhanced preprocessing with new features
    df = preprocess_data(df)
    print("Enhanced data preprocessing completed. Fraud labels removed for unsupervised learning.")
    print("New features created: IS_WEEKEND, IS_NIGHT, IS_NEW_SUBSCRIBER, RECENT_SIM_CHANGE, DURATION_TO_CHARGE_RATIO, HIGH_COST_SHORT_CALL")
    
    # Create proxy labels for supervised comparison
    df['ANOMALY_SCORE'] = df.apply(lambda row: 
        abs(row.get('CDR_DURATION_SEC', 0)) * 1.5 + 
        abs(row.get('CHARGE_FEE_1', 0)) * 2.0 + 
        (2.5 if row.get('IS_NIGHT', False) else 0) +
        (3.0 if row.get('RECENT_SIM_CHANGE', False) else 0) +
        (2.0 if row.get('OTHER_PARTY_RARE', False) else 0), 
        axis=1)
    
    score_75 = df['ANOMALY_SCORE'].quantile(0.75)
    score_90 = df['ANOMALY_SCORE'].quantile(0.90)
    df['ANOMALY_BASED_PREDICTION'] = pd.cut(df['ANOMALY_SCORE'], 
                                           bins=[-1, score_75, score_90, float('inf')], 
                                           labels=[0, 1, 2])
    
    y_proxy = df['ANOMALY_BASED_PREDICTION'].values
    
    print("\n" + "="*80)
    print("XGBOOST WITH CLASS IMBALANCE HANDLING")
    print("="*80)
    
    # Initialize and train XGBoost model
    xgb_detector = XGBoostFraudDetector(use_smote=True)
    xgb_fit_time, X_test, y_test, xgb_pred, xgb_proba = xgb_detector.train(df, y_proxy)
    
    # Make predictions on full dataset
    xgb_start_time = time.time()
    xgb_predictions, xgb_probabilities = xgb_detector.predict(df)
    xgb_predict_time = time.time() - xgb_start_time
    
    # Evaluate XGBoost
    xgb_results = evaluate_model_comprehensive(
        df, xgb_predictions, xgb_probabilities, "XGBoost with SMOTE", 
        xgb_fit_time, xgb_predict_time, y_proxy
    )
    
    print("\n" + "="*80)
    print("Q-LEARNING AGENT PERFORMANCE")
    print("="*80)
    
    # Calculate the number of unique states for Q-learning
    unique_states = df.apply(lambda row: (
       int(row.get('CDR_DURATION_SEC', 0) * 10),
       row.get('CALL_HOUR', 12),
       row.get('OTHER_PARTY', 0) % 100,  # Reduced cardinality
       int(row.get('CHARGE_FEE_1', 0) * 10),
       int(row.get('IS_NIGHT', False)),
       int(row.get('RECENT_SIM_CHANGE', False))
    ), axis=1).drop_duplicates().shape[0]
    
    n_actions = 3  # 0: Normal, 1: Suspicious, 2: Fraud
    
    print(f"Number of unique states: {unique_states}")
    print(f"Number of actions: {n_actions}")
    
    # Initialize enhanced unsupervised Q-learning agent with class weights
    agent = UnsupervisedQLearning(n_actions, unique_states, epsilon=0.3, epsilon_decay=0.998)
    print("Class weights for imbalance handling:", agent.class_weights)
    
    reward_log = []
    state_visit_count = defaultdict(int)
    
    # Set epochs - increased for better learning
    dataset_size = len(df)
    epochs = min(900, dataset_size)  # Increased epochs
    print(f"Number of epochs set to {epochs}")
    
    # Enhanced training loop with progress tracking
    start_time = time.time()
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
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    q_learning_fit_time = time.time() - start_time
    
    # Extract the optimal policy and get probabilities
    optimal_policy = agent.get_optimal_policy()
    
    # Apply the optimal policy to the dataset with timing
    start_time = time.time()
    df['PREDICTED_ACTION'] = df.apply(lambda row: optimal_policy[agent.get_state(row)], axis=1)
    
    # Get probabilities for ROC-AUC and PR-AUC calculation
    probabilities = []
    for idx, row in df.iterrows():
        state = agent.get_state(row)
        prob = agent.get_action_probabilities(state)
        probabilities.append(prob)
    
    probabilities = np.array(probabilities)
    q_learning_predict_time = time.time() - start_time
    
    # Evaluate Q-learning
    q_learning_results = evaluate_model_comprehensive(
        df, df['PREDICTED_ACTION'].values, probabilities, "Q-Learning Agent", 
        q_learning_fit_time, q_learning_predict_time, y_proxy
    )
    
    # Map actions to fraud types
    action_mapping = {0: 'Normal', 1: 'Subscription_Fraud', 2: 'SIM_Swap_Fraud'}
    df['PREDICTED_FRAUD_TYPE'] = df['PREDICTED_ACTION'].map(action_mapping)
    df['XGBOOST_PREDICTION'] = xgb_predictions
    df['XGBOOST_FRAUD_TYPE'] = df['XGBOOST_PREDICTION'].map(action_mapping)
    
    # Print prediction distribution comparison
    print("\n" + "="*80)
    print("PREDICTION DISTRIBUTION COMPARISON")
    print("="*80)
    print("Q-Learning Distribution:")
    print(df['PREDICTED_FRAUD_TYPE'].value_counts())
    print("\nXGBoost Distribution:")
    print(df['XGBOOST_FRAUD_TYPE'].value_counts())
    
    # Comparative visualization
    plt.figure(figsize=(15, 12))
    
    # Prediction distribution comparison
    plt.subplot(2, 2, 1)
    q_learning_counts = df['PREDICTED_FRAUD_TYPE'].value_counts()
    xgboost_counts = df['XGBOOST_FRAUD_TYPE'].value_counts()
    
    x = np.arange(len(action_mapping))
    width = 0.35
    
    plt.bar(x - width/2, [q_learning_counts.get(action_mapping[i], 0) for i in range(3)], 
            width, label='Q-Learning', alpha=0.7)
    plt.bar(x + width/2, [xgboost_counts.get(action_mapping[i], 0) for i in range(3)], 
            width, label='XGBoost', alpha=0.7)
    
    plt.xlabel('Fraud Type')
    plt.ylabel('Count')
    plt.title('Prediction Distribution Comparison')
    plt.xticks(x, ['Normal', 'Subscription_Fraud', 'SIM_Swap_Fraud'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sensitivity comparison
    plt.subplot(2, 2, 2)
    ql_sensitivity = [q_learning_results['sensitivity'].get(i, 0) for i in range(3)]
    xgb_sensitivity = [xgb_results['sensitivity'].get(i, 0) for i in range(3)]
    
    plt.bar(x - width/2, ql_sensitivity, width, label='Q-Learning', alpha=0.7)
    plt.bar(x + width/2, xgb_sensitivity, width, label='XGBoost', alpha=0.7)
    
    plt.xlabel('Class')
    plt.ylabel('Sensitivity (Recall)')
    plt.title('Sensitivity Comparison by Class')
    plt.xticks(x, ['Normal', 'Subscription_Fraud', 'SIM_Swap_Fraud'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training time comparison
    plt.subplot(2, 2, 3)
    models = ['Q-Learning', 'XGBoost']
    training_times = [q_learning_fit_time, xgb_fit_time]
    
    plt.bar(models, training_times, color=['skyblue', 'lightcoral'], alpha=0.7)
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.grid(True, alpha=0.3)
    
    # Overall performance comparison
    plt.subplot(2, 2, 4)
    overall_metrics = ['Overall Sensitivity', 'ROC-AUC', 'PR-AUC']
    ql_metrics = [
        q_learning_results['overall_sensitivity'],
        q_learning_results.get('roc_auc', {}).get('macro', 0) if q_learning_results.get('roc_auc') else 0,
        q_learning_results.get('pr_auc', {}).get('macro', 0) if q_learning_results.get('pr_auc') else 0
    ]
    xgb_metrics = [
        xgb_results['overall_sensitivity'],
        xgb_results.get('roc_auc', {}).get('macro', 0) if xgb_results.get('roc_auc') else 0,
        xgb_results.get('pr_auc', {}).get('macro', 0) if xgb_results.get('pr_auc') else 0
    ]
    
    x_metrics = np.arange(len(overall_metrics))
    plt.bar(x_metrics - width/2, ql_metrics, width, label='Q-Learning', alpha=0.7)
    plt.bar(x_metrics + width/2, xgb_metrics, width, label='XGBoost', alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Overall Performance Comparison')
    plt.xticks(x_metrics, overall_metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    #*****************************************************************************************************
    # === Visualizations for Q-learning ===
    # Average Reward per Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(reward_log) + 1), reward_log, color='blue', linewidth=2)
    plt.title('Average Reward per Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Q-Table Heatmap (First 20 States)
    plt.figure(figsize=(8, 6))
    first_20_states = agent.q_table[:20, :]
    sns.heatmap(first_20_states, annot=True, cmap='coolwarm', cbar=True,
                xticklabels=['Normal', 'Subscription_Fraud', 'SIM_Swap_Fraud'])
    plt.title('Q-Table Heatmap (First 20 States)', fontsize=14)
    plt.xlabel('Actions')
    plt.ylabel('States (First 20)')
    plt.tight_layout()
    plt.show()
    #*****************************************************************************************************
    # Save the updated dataset with both predictions
    output_columns = [col for col in df.columns if col not in ['PREDICTED_ACTION', 'XGBOOST_PREDICTION']]
    df[output_columns].to_excel("Telkom_Enhanced_Predictions_Comparison.xlsx", index=False)
    print("\nEnhanced dataset saved with both Q-Learning and XGBoost predictions")