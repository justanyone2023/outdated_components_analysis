import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error, median_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split



def load_data():
    # Load data
    df = pd.read_csv('./data/merged_with_type.csv')

    # Drop rows with NaN values
    df = df.dropna()

    # Modify 'type' column
    df['type'] = df['type'].apply(lambda x: 1 if x == 'normal' else 0)

    cols_to_check = [
        'new_contributors', 'change_request_response_time_avg', 'issue_age_avg',
        'code_change_lines_sum', 'issues_new', 'issues_and_change_request_active',
        'code_change_lines_add', 'attention', 'issue_comments', 'change_requests_accepted',
        'change_request_age_avg', 'participants', 'bus_factor', 'code_change_lines_remove',
        'inactive_contributors', 'change_requests_reviews', 'activity', 'change_request_resolution_duration_avg',
        'issues_closed', 'change_requests', 'issue_response_time_avg', 'issue_resolution_duration_avg', 'stars'
    ]

    duplicates = df.duplicated()
    if duplicates.sum() > 0:
        print(f"Found {duplicates.sum()} duplicate rows.")
        df = df[~duplicates]


    mask = (df[cols_to_check] == 0).sum(axis=1) > len(cols_to_check) / 2


    df = df[~mask]
    print(df.shape)
    y = df['type']
    X = df.drop(columns=['repo_name', 'Date', 'type'])  # 'Date' is already dropped, so this might raise a KeyError


    return X, y


def preprocess_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def train_and_evaluate_model(model_name, model, X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model.fit(X_train, y_train)

    # Predict using the model
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # For ROC-AUC
    try:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob)
    except:
        roc_auc = "Not Available"

    final_estimator = model.named_steps['model'] if isinstance(model, Pipeline) else model

    if hasattr(final_estimator, 'feature_importances_'):
        importances = final_estimator.feature_importances_
    elif hasattr(final_estimator, 'coef_'):
        importances = final_estimator.coef_[0]  # For models like Logistic Regression
    else:
        importances = None

    if importances is not None:
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        })
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        feature_importances.to_csv(f"./result/{model_name}_feature_importances.csv", index=False)
    else:
        print(f"Feature importance not available for {model_name}")
        feature_importances = pd.DataFrame(columns=['Feature', 'Importance'])

    # Store metrics in a DataFrame
    result = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    })

    return result, feature_importances  # Also return the feature_importances DataFrame

# Check class distribution
def check_class_distribution(y):
    class_counts = y.value_counts()
    print(f"Class distribution:\n{class_counts}")

def main():
    if not os.path.exists('result'):
        os.makedirs('result')

    X, y = load_data()

    # Check class distribution
    check_class_distribution(y)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', RandomForestRegressor())
    ])


    models = [
        ('Logistic Regression', LogisticRegression(max_iter=10000), {}),  # Increased max_iter for convergence
        ('Decision Tree Classifier', DecisionTreeClassifier(), {'model__max_depth': range(1, 10)}),
        ('Support Vector Machine', SVC(), {'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'model__C': [1, 10]}),
        ('K-Nearest Neighbors', KNeighborsClassifier(), {'model__n_neighbors': range(1, 10)}),
        ('Random Forest Classifier', RandomForestClassifier(), {'model__n_estimators': [100, 200, 300]}),
        ('Gradient Boosting Classifier', GradientBoostingClassifier(), {'model__learning_rate': [0.1, 0.01], 'model__n_estimators': [100, 200]}),
        ('AdaBoost Classifier', AdaBoostClassifier(), {'model__learning_rate': [0.1, 0.01], 'model__n_estimators': [100, 200]})
    ]

    results = []

    for name, model, params in models:
        pipeline.set_params(model=model)

        # Use GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(pipeline, params, scoring='accuracy', cv=KFold(n_splits=5, shuffle=True, random_state=42))
        grid_search.fit(X, y)

        # Get the best model and evaluate it
        best_model = grid_search.best_estimator_
        metrics_df, feature_importances = train_and_evaluate_model(name, best_model, X, y)

        # Save the evaluation metrics to a CSV file
        metrics_df.to_csv(f"./result/{name}_metrics.csv", index=False)

        # Save feature importance to a CSV file if available
        if not isinstance(feature_importances, str):  # If it's not the "not available" string
            feature_importances.to_csv(f"./result/{name}_feature_importances.csv", index=False)

if __name__ == '__main__':
    main()
