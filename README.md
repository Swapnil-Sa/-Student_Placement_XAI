# 🎓 Student Placement Prediction using Explainable AI (XAI)

This project predicts whether a student will be **Placed** or **Not Placed** using academic scores, skills, certifications, internships, and training data.  
It also uses **Explainable AI (XAI)** techniques like **SHAP** to explain how each feature influences the model’s predictions.

---

## 🚀 Project Objective

The goal of this project is to:
- Predict student placement outcomes using Machine Learning  
- Improve prediction accuracy using **SMOTE**, **scaling**, and **CatBoost**  
- Make the model explainable and transparent using **SHAP**  
- Identify which student attributes most strongly affect placement chances  

---

# 📂 Dataset Information

The dataset contains **student academic, skill, and training information**.  
Below are the features:

| Column | Description |
|--------|-------------|
| `CGPA` | Student’s Grade Point Average |
| `Internships` | Number of internships completed |
| `Projects` | Number of technical/academic projects |
| `Workshops/Certifications` | Training or certification count |
| `AptitudeTestScore` | Aptitude & reasoning score |
| `SoftSkillsRating` | Communication & interpersonal skills rating |
| `ExtracurricularActivities` | Participation in extracurricular activities (Yes/No) |
| `PlacementTraining` | Whether student attended placement training (Yes/No) |
| `SSC_Marks` | 10th board marks |
| `HSC_Marks` | 12th board marks |
| `PlacementStatus` | Target variable → Placed / NotPlaced |

### ✔ Dataset Actions Performed in Code
- Loaded dataset and displayed the first few rows  
- Checked & handled missing values  
- Encoded categorical features (Yes/No → 1/0)  
- Converted `PlacementStatus` into binary form (Placed=1, NotPlaced=0)  
- Analyzed distribution of features  
- Verified shape, null values, and balance of target column  

---

# 🧠 Machine Learning Workflow (Exactly as in the Code)

## 1️⃣ Importing Libraries
The notebook uses:
- **Pandas, NumPy** for data handling  
- **sklearn** for preprocessing, scaling, train-test split  
- **SMOTE** from imblearn to balance data  
- **CatBoostClassifier** for modeling  
- **SHAP** for explainability  
- **Matplotlib/Seaborn** for plots  

---

## 2️⃣ Data Preprocessing

✔ Encoded all categorical columns  
✔ Encoded target variable `PlacementStatus` into numerical form  
✔ Separated dataset into:
- **X** → features  
- **y** → target  

✔ Checked shapes and null values  
✔ Scaled continuous features using **StandardScaler**  

---
## 3️⃣ Key Results

✔ CatBoost achieved high prediction accuracy

✔ SMOTE successfully balanced the dataset and improved fairness

✔ Scaling optimized model behavior

✔ SHAP revealed the most important features influencing placement

## 4️⃣ Top Influential Features (from SHAP)

- Aptitude Test Score – Higher aptitude increases placement chance

- CGPA – Strong academic performance is a key factor

- Soft Skills Rating – Communication & teamwork influence selection

- Internships – Practical experience boosts placement likelihood

- Workshops/Certifications – Technical add-ons have strong weightage

- Projects – More hands-on project work helps placement chances

- These features together have the highest contribution in predicting placement outcomes.


## 5️⃣ Handling Imbalance (SMOTE)

The dataset is imbalanced (more NotPlaced than Placed).  
To fix this:

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)


