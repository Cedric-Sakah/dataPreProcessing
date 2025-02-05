import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset from Seaborn
df = sns.load_dataset('titanic')
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Drop rows with any missing values
df_dropped_rows = df.dropna()
print("DataFrame after dropping rows with missing values:\n", df_dropped_rows)

# Drop columns with any missing values
df_dropped_columns = df.dropna(axis=1)
print("DataFrame after dropping columns with missing values:\n", df_dropped_columns)

# Mean imputation for numerical columns
df['age'].fillna(df['age'].mean(), inplace=True)

# Median imputation for numerical columns
df['age'].fillna(df['age'].median(), inplace=True)

# Mode imputation for categorical columns
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Check for any remaining missing values
print("Missing values after imputation:\n", df.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Convert categorical columns to numerical for correlation analysis
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Display correlation with target variable 'survived'
print("Correlation with 'survived':\n", correlation_matrix['survived'].sort_values(ascending=False))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Select relevant columns and drop rows with missing values for simplicity
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'sibsp', 'parch', 'embarked', 'family_size']].dropna()

# Split features and target variable
X = df.drop('survived', axis=1)
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
print("Feature Importances:\n", feature_importances)

# Plot feature importances
feature_importances.plot(kind='bar', figsize=(10, 6), title="Feature Importance from Random Forest")
plt.show()
