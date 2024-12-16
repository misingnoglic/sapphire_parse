import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import graphviz
from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import export_graphviz
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("sapphires.csv")
df["length_width_ratio"] = df["Length"]/df["Width"]

# Remove colums that are not useful for the model
# All columns: ['name', 'url', 'image_url', 'Item ID', 'Total Price', 'Weight',
#        'Per Carat Price', 'Color', 'Shape', 'Clarity', 'Cut',
#        'Color Intensity', 'Origin', 'Treatments', 'Length', 'Width', 'Height',
#        'Price per Length', 'length_width_ratio']
# df["Per Carat Price"] = df['Weight'] / df['Total Price']
df = df.drop(columns=["name", "url", "image_url", "Item ID"])

# Drop any columns related to price
df = df.drop(columns=["Per Carat Price", "Price per Length"])

# Drop ones where color is padparadscha
# df = df[df["Color"] != "Padparadscha (Pinkish-Orange / Orangish-Pink)"]

# Define features and target
X = df.drop('Total Price', axis=1)
y = df['Total Price']

categorical_cols = ['Color', 'Shape', 'Clarity', 'Cut', 'Color Intensity', 'Origin', 'Treatments']
numerical_cols = ['Weight', 'Length', 'Width', 'Height', 'length_width_ratio']

# Preprocessing: OneHotEncoding for categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Optional: Feature importance for interpretability
feature_names = (
        numerical_cols +
        list(model.named_steps['preprocessor']
             .named_transformers_['cat']
             .get_feature_names_out(categorical_cols))
)

feature_importances = model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))
