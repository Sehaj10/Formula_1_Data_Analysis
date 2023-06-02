import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the telemetry data into a pandas DataFrame
data = pd.read_csv("Telemetry_Austria.csv")

# Select the relevant features for analysis
features = [
    "trackLength",
    "lap_distance",
    "Velocity kmph",
    "lap_time",
    #"throttle",
    #"brake",
    "steering",
    "rpm",
    "world_position_X",
    "world_position_Y",

    
    # Add more relevant features as needed
]

# Select the target variable
target = "lap_time"

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)

# Create an empty dictionary to store the model predictions
predictions = {}

# Initialize and train the linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions using the linear regression model
linear_reg_preds = linear_reg.predict(X_test)
predictions["Linear Regression"] = linear_reg_preds

# Initialize and train the decision tree regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

# Make predictions using the decision tree regressor
tree_reg_preds = tree_reg.predict(X_test)
predictions["Decision Tree"] = tree_reg_preds

# Initialize and train the random forest regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

# Make predictions using the random forest regressor
forest_reg_preds = forest_reg.predict(X_test)
predictions["Random Forest"] = forest_reg_preds

# Compare the performance of the models using evaluation metrics
for model, preds in predictions.items():
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{model} MSE: {mse}")
    print(f"{model} MAE: {mae}")
    print()

# Calculate the lap time predictions for a new set of data
new_data = pd.read_csv("Telemetry_Austria.csv")  # Replace with your new data file
new_preds = forest_reg.predict(new_data[features])
print("New data lap time predictions:")
print(new_preds)
