import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
def train_flood_risk_model():
    # Data preparation (Original and synthetic data)
    data = pd.DataFrame({
        'water_level': [100, 150, 200, 250, 300, 350, 400],
        'rainfall': [10, 20, 30, 40, 50, 60, 70],
        'flood_risk': [0, 0, 1, 1, 1, 1, 1]
    })
    synthetic_data = pd.DataFrame({
        'water_level': np.random.uniform(50, 500, 300),
        'rainfall': np.random.uniform(5, 150, 300)
    })
    synthetic_data['flood_risk'] = synthetic_data.apply(
        lambda row: 1 if row['water_level'] > 300 and row['rainfall'] > 50 else 0, axis=1
    )
    complete_data = pd.concat([data, synthetic_data], ignore_index=True)
    X = complete_data[['water_level', 'rainfall']]
    y = complete_data['flood_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, 'flood_risk_predictor.joblib')
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.show()
    plt.figure(figsize=(8, 6))
    x_min, x_max = X_train['water_level'].min() - 10, X_train['water_level'].max() + 10
    y_min, y_max = X_train['rainfall'].min() - 10, X_train['rainfall'].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    scatter = plt.scatter(X_test['water_level'], X_test['rainfall'], c=y_test, cmap='coolwarm', edgecolor='k')
    plt.colorbar(scatter)
    plt.title('Decision Boundary and Test Data Points')
    plt.xlabel('Water Level')
    plt.ylabel('Rainfall')
    plt.show()
if __name__ == '__main__':
    train_flood_risk_model()







