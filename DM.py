import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Student Academic Performance Prediction")
print("=" * 60)

try:
    df = pd.read_csv("student-mat.csv", sep=";")
    print(f"\n[✓] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())
except FileNotFoundError:
    print("\n[!] student-mat.csv not found. Please place it in the same folder as DM.py")
    exit()

print("\n──────────── DATA PREPROCESSING ────────────")

df["result"] = df["G3"].apply(lambda g: "Pass" if g >= 10 else "Fail")
print(f"\nClass distribution:\n{df['result'].value_counts().to_string()}")

missing = df.isnull().sum().sum()
print(f"\nMissing values: {missing}")
if missing > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include="object"):
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("  → Filled with median/mode.")

le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("result")
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df["result_encoded"] = le.fit_transform(df["result"])

X = df.drop(columns=["G3", "result", "result_encoded"])
y = df["result_encoded"]

print(f"\nFeatures used: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set : {X_train.shape[0]} samples")
print(f"Testing set  : {X_test.shape[0]} samples")

print("\n──────────── MODEL TRAINING ────────────")
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "class_weight": "balanced",
}
for k, v in params.items():
    print(f"{k:22s} = {v}")

model = RandomForestClassifier(**params)
model.fit(X_train, y_train)
print("\n[✓] Training complete.")

print("\n──────────── RESULTS & ANALYSIS ────────────")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})
COLORS = ["#E74C3C", "#2ECC71"]

fig, ax = plt.subplots(figsize=(5, 4))
counts = df["result"].value_counts()
ax.bar(counts.index, counts.values, color=COLORS, edgecolor="white", linewidth=1.5)
ax.set_title("Fig 1 – Class Distribution (Pass / Fail)", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Students")
for i, v in enumerate(counts.values):
    ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("fig1_class_distribution.png")
plt.close()

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Fig 2 – Confusion Matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("fig2_confusion_matrix.png")
plt.close()

importances = pd.Series(model.feature_importances_, index=X.columns)
top12 = importances.nlargest(12).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
top12.plot(kind="barh", ax=ax, color="#3498DB", edgecolor="white")
ax.set_title("Fig 3 – Top 12 Feature Importances", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("fig3_feature_importance.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
study_pass = df.groupby("studytime")["result"].apply(
    lambda x: (x == "Pass").sum() / len(x) * 100
)
ax.bar(study_pass.index, study_pass.values, color="#9B59B6", edgecolor="white", linewidth=1.5)
ax.set_title("Fig 4 – Study Time vs Pass Rate (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Study Time (1=<2h  2=2-5h  3=5-10h  4=>10h)")
ax.set_ylabel("Pass Rate (%)")
ax.set_ylim(0, 110)
for i, v in zip(study_pass.index, study_pass.values):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=9)
plt.tight_layout()
plt.savefig("fig4_studytime_passrate.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
bins = range(0, 35, 2)
fail_data = df[df["result"] == "Fail"]["absences"]
pass_data = df[df["result"] == "Pass"]["absences"]
ax.hist(fail_data, bins=bins, weights=[100/len(fail_data)]*len(fail_data),
        label="Fail", color="#E74C3C", edgecolor="white", alpha=0.85, rwidth=0.45,
        align="left")
ax.hist(pass_data, bins=bins, weights=[100/len(pass_data)]*len(pass_data),
        label="Pass", color="#2ECC71", edgecolor="white", alpha=0.85, rwidth=0.45,
        align="mid")
ax.set_title("Fig 5 – Absences Distribution by Result (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Absences")
ax.set_ylabel("Percentage of Students in Group (%)")
ax.legend()
plt.tight_layout()
plt.savefig("fig5_absences_distribution.png")
plt.close()

print("\n══════════════════════════════════════════")
print("  All outputs saved to current folder")
print("══════════════════════════════════════════")
