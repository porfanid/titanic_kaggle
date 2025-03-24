import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel

# Load data
train_data = pd.read_csv('./titanic/train.csv')
test_data = pd.read_csv('./titanic/test.csv')


# Feature Engineering
def process_data(df):
    # Extract titles from names - fix escape sequence
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Major': 'Officer',
        'Col': 'Officer',
        'Capt': 'Officer',
        'Jonkheer': 'Royalty',
        'Don': 'Royalty',
        'Sir': 'Royalty',
        'Lady': 'Royalty',
        'the Countess': 'Royalty',
        'Dona': 'Royalty',
        'Mme': 'Mrs',
        'Mlle': 'Miss',
        'Ms': 'Mrs'
    }
    df['Title'] = df['Title'].map(lambda x: title_mapping.get(x, 'Other'))

    # Create family size and is_alone features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Create age groups for better segmentation
    # Handle NaN values before creating age bands
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 40, 60, 100],
                           labels=['Child', 'Teenager', 'Adult', 'Senior', 'Elderly'])

    # Extract cabin deck information when available
    df['Deck'] = df['Cabin'].str.slice(0, 1)
    df['Deck'] = df['Deck'].fillna('U')  # U for Unknown

    # Extract fare bands - handle NaN values first
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareBand'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'],
                             duplicates='drop')

    # Create a new feature combining class and sex
    df['Class_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']

    # Process ticket information (extracting first letter/number)
    df['TicketPrefix'] = df['Ticket'].str.split().apply(lambda x: x[0] if len(x) > 1 else 'X')

    # Extract ticket number if it exists - fix escape sequence
    df['TicketNum'] = df['Ticket'].str.extract(r'(\d+)', expand=False)
    df['TicketNum'] = pd.to_numeric(df['TicketNum'], errors='coerce')
    df['TicketNum'] = df['TicketNum'].fillna(0)  # Replace NaN with 0

    # Use TF-IDF on combined text data (Name + Ticket + Cabin when available)
    df['TextData'] = df['Name'] + ' ' + df['Ticket']
    df.loc[df['Cabin'].notna(), 'TextData'] += ' ' + df['Cabin']

    return df


# Process both datasets
train_data = process_data(train_data)
test_data = process_data(test_data)

# Prepare the TF-IDF vectorizer for text data
tfidf = TfidfVectorizer(min_df=3, max_features=100,
                        ngram_range=(1, 2), sublinear_tf=True)

# Fit and transform on train data
tfidf_train = tfidf.fit_transform(train_data['TextData'].fillna(''))
tfidf_features = pd.DataFrame(tfidf_train.toarray(),
                              columns=[f'tfidf_{i}' for i in range(tfidf_train.shape[1])])

# Transform test data
tfidf_test = tfidf.transform(test_data['TextData'].fillna(''))
tfidf_test_features = pd.DataFrame(tfidf_test.toarray(),
                                   columns=[f'tfidf_{i}' for i in range(tfidf_test.shape[1])])

# Add the TF-IDF features to the datasets
train_data = pd.concat([train_data.reset_index(drop=True),
                        tfidf_features.reset_index(drop=True)], axis=1)
test_data = pd.concat([test_data.reset_index(drop=True),
                       tfidf_test_features.reset_index(drop=True)], axis=1)

# Fixed: Properly handle categorical columns
# Replace fillna with direct assignment to avoid categorical data issues
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Handle categorical variables - with fix for categorical data
categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeBand', 'Deck', 'Class_Sex', 'FareBand']
for col in categorical_cols:
    if col in train_data.columns and col in test_data.columns:
        # Convert to string first to handle categorical data properly
        train_col_str = train_data[col].astype(str)
        test_col_str = test_data[col].astype(str)

        le = LabelEncoder()
        le.fit(list(train_col_str.values) + ['Missing'])  # Include 'Missing' in the fit

        train_data[f'{col}_encoded'] = le.transform(train_col_str)
        test_data[f'{col}_encoded'] = le.transform(test_col_str)

# Feature scaling for numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'FamilySize', 'TicketNum']
for col in numerical_cols:
    if col in train_data.columns and col in test_data.columns:
        train_data[f'{col}_scaled'] = scaler.fit_transform(train_data[[col]])
        test_data[f'{col}_scaled'] = scaler.transform(test_data[[col]])

# Select features for model training
feature_cols = [col for col in train_data.columns if col.endswith('_encoded')
                or col.endswith('_scaled') or col.startswith('tfidf_')
                or col in ['IsAlone', 'Pclass']]

# Prepare data for model training
X = train_data[feature_cols]
y = train_data['Survived']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a base model for feature selection
feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
feature_selector.fit(X_train, y_train)

# Select the most important features
selector = SelectFromModel(feature_selector, threshold='median', prefit=True)
feature_indices = selector.get_support()
selected_features = X.columns[feature_indices]
print(f"Selected {len(selected_features)} features: {', '.join(selected_features.tolist()[:10])}...")

X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)


# Model Training with multiple algorithms
def train_evaluate_model(model_name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"{model_name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    print(classification_report(y_val, y_pred))

    return model, auc, accuracy


# Random Forest with improved hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42
)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

# Train and evaluate both models
rf_trained, rf_auc, rf_acc = train_evaluate_model(
    "Random Forest", rf_model, X_train_selected, y_train, X_val_selected, y_val
)

gb_trained, gb_auc, gb_acc = train_evaluate_model(
    "Gradient Boosting", gb_model, X_train_selected, y_train, X_val_selected, y_val
)

# Choose the best model based on AUC
if rf_auc > gb_auc:
    best_model = rf_trained
    print("Using Random Forest as the final model")
else:
    best_model = gb_trained
    print("Using Gradient Boosting as the final model")

# Prepare final test data for prediction
X_test = test_data[feature_cols]
X_test_selected = selector.transform(X_test)

# Generate predictions
test_data['Survived'] = best_model.predict(X_test_selected).astype(int)

# Generate the submission file
submission = test_data[['PassengerId', 'Survived']]
submission.to_csv('improved_submission.csv', index=False)
print("Submission file created: improved_submission.csv")

# Feature importance analysis
if isinstance(best_model, RandomForestClassifier):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature ranking:")
    selected_feature_names = list(selected_features)
    for f in range(min(10, len(selected_feature_names))):
        idx = indices[f]
        if idx < len(selected_feature_names):
            print(f"{f + 1}. {selected_feature_names[idx]} ({importances[idx]:.4f})")