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
Frontend Code:
<form onsubmit="event.preventDefault(); predictFloodRisk();">
    <label for="waterLevel">Water Level (cm):</label>
    <input type="number" id="waterLevel" name="waterLevel" step="0.1" required>
    <label for="rainfall">Rainfall (mm):</label>
    <input type="number" id="rainfall" name="rainfall" step="0.1" required>
    <label for="location">Location:</label>
    <input type="text" id="location" name="location" required>
    <button type="submit">Predict</button>
</form>
Backend (Flask) Code:
@app.route('/predictFloodRisk', methods=['POST'])
def predict_flood_risk():
    data = request.get_json()
    water_level = data.get('water_level')
    rainfall = data.get('rainfall')
    location = data.get('location', 'Unknown')
    flood_risk = model.predict([[float(water_level), float(rainfall)]])[0]
    risk_label = "High" if flood_risk else "Low"
    # Store the prediction in the database
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO water_data (water_level, rainfall, location, risk_level) VALUES (%s, %s, %s, %s)",
        (water_level, rainfall, location, risk_label)
    )
    mysql.connection.commit()
    cur.close()
    return jsonify({
        "water_level": water_level,
        "rainfall": rainfall,
        "location": location,
        "flood_risk": risk_label
    })
Database Schema:
CREATE TABLE water_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    water_level FLOAT NOT NULL,
    rainfall FLOAT NOT NULL,
    location VARCHAR(255) NOT NULL,
    risk_level VARCHAR(50), -- High/Low risk from model
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);








