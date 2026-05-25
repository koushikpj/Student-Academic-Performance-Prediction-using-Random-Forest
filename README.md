# Student Academic Performance Prediction using Random Forest  

---

##  Project Overview
This project applies **Data Mining techniques** to predict student academic performance (Pass/Fail) using the **UCI Student Performance Dataset**.  
The model is built with a **Random Forest Classifier**, providing insights into factors that influence student success such as study time, absences, and prior grades.

---

##  Objectives
- Predict whether a student will **Pass** or **Fail** based on available features.  
- Analyze key factors influencing academic performance.  
- Provide visual insights for educators and researchers.  

---

##  Dataset
- **Source:** [UCI Machine Learning Repository – Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)  
- **File Used:** `student-mat.csv`  
- **Features:** Demographics, family background, study time, absences, grades (G1, G2, G3).  
- **Target:** Binary classification → `Pass` (G3 ≥ 10) / `Fail` (G3 < 10).  

---

##  Methodology
1. **Data Preprocessing**
   - Handle missing values (median/mode imputation).  
   - Encode categorical variables using Label Encoding.  
   - Create binary target variable (`Pass`/`Fail`).  

2. **Model Training**
   - Algorithm: **Random Forest Classifier**  
   - Parameters:  
     - `n_estimators = 100`  
     - `max_depth = 10`  
     - `min_samples_split = 5`  
     - `min_samples_leaf = 2`  
     - `class_weight = balanced`  

3. **Evaluation**
   - Accuracy Score  
   - Classification Report (Precision, Recall, F1)  
   - Confusion Matrix  

4. **Visualisations**
   - Class Distribution (Pass vs Fail)  
   - Confusion Matrix Plot  
   - Top 12 Feature Importances  
   - Study Time vs Pass Rate  
   - Absences Distribution  

---

##  Results
- Achieved strong classification accuracy with Random Forest.  
- **Key Insights:**
  - **G2 (second period grade)** is highly predictive of final outcome.  
  - **Study time** positively correlates with pass rate.  
  - **Absences** strongly linked to failure.  

---

## 👨‍💻 Team Members
- Koushik PJ – 23ETCS002062  
- Mohammed Saad – 23ETCS002073  
- Pathikrit Datta – 23ETCS002084  
- Pranay Parekh – 23ETCS002091  

---

##  Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  

---

## 📌 How to Run
```bash
# Clone the repository
git clone https://github.com/koushikpj/Student-Academic-Performance-Prediction-using-Random-Forest.git

# Navigate to project folder
cd Student-Academic-Performance-Prediction-using-Random-Forest

# Run the script
python DM.py
