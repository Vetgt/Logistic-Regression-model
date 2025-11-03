import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'diabetes_012_health_indicators_BRFSS2015.csv'
df = pd.read_csv(file_path)

print("=== FIRST 5 ROWS ===")
print(df.head(), "\n")

df.dropna(inplace=True)
df = df.astype(float)

df['diabetes_binary'] = df['Diabetes_012'].apply(lambda x: 0 if x == 0 else 1)

X = df.drop(['Diabetes_012', 'diabetes_binary'], axis=1)
y = df['diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("=== MODEL PERFORMANCE ===")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
labels = ['Non-Diabetes', 'Diabetes']

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("\n=== MANUAL TEST FOR USER (INPUT DATA) ===")

try:
    sample = {}

    sample['HighBP'] = 1 if input("Do you have high blood pressure? (yes/no): ").lower() == 'yes' else 0
    sample['HighChol'] = 1 if input("Do you have high cholesterol? (yes/no): ").lower() == 'yes' else 0
    sample['CholCheck'] = 1 if input("Have you checked your cholesterol in the last 5 years? (yes/no): ").lower() == 'yes' else 0
    sample['BMI'] = float(input("What is your BMI value (e.g., 22.5): "))
    sample['Smoker'] = 1 if input("Are you a current smoker? (yes/no): ").lower() == 'yes' else 0
    sample['Stroke'] = 1 if input("Have you ever had a stroke? (yes/no): ").lower() == 'yes' else 0
    sample['HeartDiseaseorAttack'] = 1 if input("Have you ever had a heart attack? (yes/no): ").lower() == 'yes' else 0
    sample['PhysActivity'] = 1 if input("Do you regularly exercise? (yes/no): ").lower() == 'yes' else 0
    sample['Fruits'] = 1 if input("Do you eat fruits daily? (yes/no): ").lower() == 'yes' else 0
    sample['Veggies'] = 1 if input("Do you eat vegetables daily? (yes/no): ").lower() == 'yes' else 0
    sample['HvyAlcoholConsump'] = 1 if input("Do you often consume heavy alcohol? (yes/no): ").lower() == 'yes' else 0
    sample['AnyHealthcare'] = 1 if input("Do you have access to healthcare services? (yes/no): ").lower() == 'yes' else 0
    sample['NoDocbcCost'] = 1 if input("Have you ever skipped a doctor visit due to cost? (yes/no): ").lower() == 'yes' else 0
    sample['GenHlth'] = float(input("How do you rate your general health? (1=Excellent, 5=Poor): "))
    sample['MentHlth'] = float(input("How many days in the past 30 days did you feel mentally unwell? (0-30): "))
    sample['PhysHlth'] = float(input("How many days in the past 30 days did you feel physically unwell? (0-30): "))
    sample['DiffWalk'] = 1 if input("Do you have difficulty walking or climbing stairs? (yes/no): ").lower() == 'yes' else 0
    sample['Sex'] = 1 if input("Your gender? (male/female): ").lower() == 'male' else 0
    sample['Age'] = float(input("Your age group (1=18-24, 2=25-29, ..., 13=80+): "))
    sample['Education'] = float(input("Your education level (1=no schooling, 6=university): "))
    sample['Income'] = float(input("Your income level (1=lowest, 8=highest): "))

    sample_df = pd.DataFrame([sample])

    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][prediction]

    print("\n=== PREDICTION RESULT ===")
    if prediction == 1:
        print(f"⚠️ You are at RISK of having Diabetes (probability {prob*100:.2f}%)")
    else:
        print(f"✅ You are NOT at risk of having Diabetes (probability {prob*100:.2f}%)")

except Exception as e:
    print("\nInput error:", e)
