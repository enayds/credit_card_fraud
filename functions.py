import numpy as np
import joblib

# Helper function for one-hot encoding
def one_hot_encode(category, day_of_week):
    # Load the encoder object
    encoder = joblib.load('onehot_encoder.pkl')
    
    # Ensure the inputs are in the correct format for encoding (as list of lists)
    # The category and day_of_week should be in a list of lists format
    encoded_data = encoder.transform([[category, day_of_week]])  # Correct format: a list containing a list
     # Same as above
    return encoded_data

    # Return the concatenated result (encoded category + day_of_week)
    # return np.concatenate([encoded_category.toarray(), encoded_day_of_week.toarray()], axis=1)


# Main function to preprocess input
def preprocess_input(transaction_amount, transaction_hour, category, gender, day, age):
    """Preprocess user inputs into model-ready format."""
    category_mapping = {
    'Misc (Net)': 'misc_net',
    'Grocery (POS)': 'grocery_pos',
    'Entertainment': 'entertainment',
    'Gas & Transport': 'gas_transport',
    'Misc (POS)': 'misc_pos',
    'Grocery (Net)': 'grocery_net',
    'Shopping (Net)': 'shopping_net',
    'Shopping (POS)': 'shopping_pos',
    'Food & Dining': 'food_dining',
    'Personal Care': 'personal_care',
    'Health & Fitness': 'health_fitness',
    'Travel': 'travel',
    'Kids & Pets': 'kids_pets',
    'Home': 'home'
}

    

    
    # One-hot encode 'category' and 'day_of_week'
    encoded_category_day = one_hot_encode(category_mapping[category], day)
    
    # Initialize feature vector (this includes continuous features and the encoded category/day features)
    features = {
        'amt': transaction_amount,
        'trans_hour': transaction_hour,
        'gender': 1 if gender == "Male" else 0,
        'age': age
    }
    
    # Create feature array by combining numerical and one-hot encoded features
    feature_array = np.array([
        features['amt'], 
        features['trans_hour'], 
        features['gender'], 
        features['age']
    ])
    
    # Append one-hot encoded features to the feature array
    feature_array = np.concatenate([feature_array, encoded_category_day.flatten()])
    
    # Reshape to ensure it is in the right format for the model
    feature_array = feature_array.reshape(1, -1)
    
    # Load scaler and transform features
    scaler = joblib.load('scaler.pkl')  # Load your saved StandardScaler
    scaled_features = scaler.transform(feature_array)
    
    return scaled_features

def is_suspicious_transaction(age, transaction_amount, transaction_hour):
    """
    Flags transactions as suspicious based on age, amount, and time of transaction.
    
    Args:
        age (int): The age of the customer.
        transaction_amount (float): The transaction amount.
        transaction_hour (int): The hour the transaction occurred (0-23).
        
    Returns:
        int: 1 if the transaction is flagged as suspicious, otherwise 0.
    """
    # Define thresholds
    high_age_threshold = 60
    high_amount_threshold = 5000
    early_morning_hours = range(0, 6)  # 12 AM to 5 AM
    late_night_hours = range(23, 24)   # 11 PM to 12 AM

    # Check if all conditions are met
    if (
        age > high_age_threshold and
        transaction_amount > high_amount_threshold and
        (transaction_hour in early_morning_hours or transaction_hour in late_night_hours)
    ):
        return 1  # Suspicious transaction
    return 0  # Not suspicious

