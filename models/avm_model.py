import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Function to load and preprocess the dataset
def load_data():
    data = pd.read_csv("data/dataset.csv")  # This assumes the dataset is downloaded and cleaned
    data = data.dropna()  # Drop missing values for simplicity
    return data

# Function to train the model
def train_model():
    data = load_data()
    
    # Define features and target
    features = ['location', 'area', 'rooms', 'year_built', 'renovation_level']
    target = 'price'
    
    X = data[features]
    y = data[target]
    
    # Convert categorical features to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Evaluation (MAE): {mae:.2f}")
    
    return model

# Function to predict price based on user input
def predict_price(model, location, area, rooms, year_built, renovation_level):
    # Convert inputs into a format suitable for prediction
    input_data = pd.DataFrame({
        'location': [location],
        'area': [area],
        'rooms': [rooms],
        'year_built': [year_built],
        'renovation_level': [renovation_level]
    })
    
    # Apply the same preprocessing steps as in training
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Ensure input matches training data features
    input_data = input_data.reindex(columns=model.get_booster().feature_names, fill_value=0)
    
    # Predict the price
    price = model.predict(input_data)
    return price[0]
