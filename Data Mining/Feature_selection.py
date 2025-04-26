from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
combined_data = pd.read_csv('data_frame/train.csv')

# Fill missing values for numerical features
combined_data['Age'] = combined_data['Age'].fillna(combined_data['Age'].mean())

# Fill missing values for categorical features
combined_data['Cabin'] = combined_data['Cabin'].fillna(combined_data['Cabin'].mode()[0])  # Use mode for categorical

# Encode categorical features
label_encoder = LabelEncoder()
combined_data['Sex'] = label_encoder.fit_transform(combined_data['Sex'])
combined_data['Embarked'] = label_encoder.fit_transform(combined_data['Embarked'])
combined_data['Cabin'] = label_encoder.fit_transform(combined_data['Cabin'])
combined_data['Ticket'] = label_encoder.fit_transform(combined_data['Ticket'])

# Check for any remaining missing values
print("Missing values after filling:")
print(combined_data.isnull().sum())

# Select features, excluding non-predictive columns
X = combined_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Cabin', 'Embarked']]
y = combined_data['Survived']

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=10)
rf_model.fit(X, y)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

print("\n\nFeatures importances")
print(feature_importances_df)

# Select top 5 features
top_5_features = feature_importances_df.head(4)
print("\n\nTop 4 features:")
print(top_5_features)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns, filled=True, fontsize=6, rounded=True)
plt.show()



