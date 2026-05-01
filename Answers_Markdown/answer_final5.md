# Introduction to Data Science — Final Exam Paper 5 (Medium)

**Course Code:** BCSE2C05 | **Semester:** II | **Max Marks:** 100 | **Time:** 3 Hours
**Difficulty:** ★★★☆☆ Medium

---

## SECTION A — Short Answer (5 Marks Each, Do Any 4 of 5 = 20 Marks)

---

### Q1. What is Data Normalization? Compare Min-Max and Z-Score normalization.

**Data Normalization** scales features to a common range so no single feature dominates in distance-based algorithms.

| Aspect | Min-Max Normalization | Z-Score (Standardization) |
| :--- | :--- | :--- |
| **Formula** | (X − Xmin) / (Xmax − Xmin) | (X − μ) / σ |
| **Range** | [0, 1] | No fixed range (centered at 0) |
| **Sensitive to outliers** | Yes (min/max affected by outliers) | Less sensitive |
| **Best for** | When data has known bounded range | When data follows normal distribution |

**Example (Data: 10, 20, 30):**
- Min-Max: 10→0.0, 20→0.5, 30→1.0
- Z-Score: μ=20, σ=8.16 → 10→−1.22, 20→0.0, 30→1.22

---

### Q2. Differentiate between Primary and Secondary data with examples.

| Aspect | Primary Data | Secondary Data |
| :--- | :--- | :--- |
| **Definition** | Collected first-hand by the researcher. | Pre-existing data collected by someone else. |
| **Source** | Surveys, experiments, interviews. | Reports, journals, databases, government publications. |
| **Cost** | Expensive and time-consuming. | Cheaper and readily available. |
| **Accuracy** | High (specific to research). | May not perfectly fit current need. |
| **Example** | Conducting a customer survey for a new product launch. | Using IRCTC booking data to analyse travel patterns. |

---

### Q3. Write Python code using NumPy to create an array, reshape it, and compute statistics.

```python
import numpy as np

# Create array
arr = np.array([12, 15, 18, 21, 24, 27, 30, 33, 36])

# Reshape to 3x3 matrix
matrix = arr.reshape(3, 3)
print("Matrix:\n", matrix)

# Statistics
print("Mean:", np.mean(arr))         # 24.0
print("Median:", np.median(arr))     # 24.0
print("Std Dev:", np.std(arr))       # 7.48
print("Variance:", np.var(arr))      # 56.0
print("Max:", np.max(arr))           # 36
print("Argmax:", np.argmax(arr))     # Index of max = 8
print("Sum of each column:", np.sum(matrix, axis=0))  # [51, 57, 63]
print("Sum of each row:", np.sum(matrix, axis=1))      # [45, 72, 99]
```

---

### Q4. What are Symmetric and Asymmetric binary attributes? Give examples.

**Binary Attributes** have exactly two values (0 and 1).

**Symmetric:** Both values carry equal importance.
- **Example:** Gender (Male/Female) — both values are equally significant.
- Similarity: Both 0-0 and 1-1 matches are counted.

**Asymmetric:** One value (usually 1) is more important or rarer.
- **Example:** Disease Test (Positive=1, Negative=0) — a positive result is much more significant.
- Similarity: Only 1-1 matches count; 0-0 matches are ignored.

**Jaccard Coefficient** (for asymmetric binary):
```
J(A, B) = f₁₁ / (f₀₁ + f₁₀ + f₁₁)
```

**Example:**

| | Patient B=1 | Patient B=0 |
|---|---|---|
| **Patient A=1** | f₁₁ = 2 | f₁₀ = 1 |
| **Patient A=0** | f₀₁ = 1 | f₀₀ = 4 |

J = 2 / (1 + 1 + 2) = 2/4 = **0.5**

---

### Q5. What are the different Data Collection Strategies? Explain any three.

| Strategy | Description | Example |
| :--- | :--- | :--- |
| **Surveys** | Structured questionnaires distributed to respondents. | Google Forms survey on student satisfaction. |
| **Web Scraping** | Automated extraction of data from websites. | Scraping Flipkart for product prices using BeautifulSoup. |
| **APIs** | Accessing data programmatically from services. | Using Twitter API to collect tweets for sentiment analysis. |
| **Observations** | Recording data by watching events. | Counting footfall in a mall at different times. |
| **Experiments** | Controlled tests with treatment and control groups. | A/B testing a website's checkout flow. |

---

## SECTION B — Analytical Questions (8 Marks Each, Do Any 5 of 6 = 40 Marks)

---

### Q6. Explain the Data Science Lifecycle. Apply it to a real-world scenario of your choice.

**Lifecycle Phases applied to "Predicting Student Exam Performance":**

| Phase | Application |
| :--- | :--- |
| **1. Problem Definition** | "Can we predict students at risk of failing based on early indicators?" |
| **2. Data Collection** | Attendance records, mid-term marks, assignment scores, LMS engagement from university database. |
| **3. Data Cleaning** | Fill missing attendance with 0 (assumed absent), remove duplicate entries, standardize date formats. |
| **4. EDA** | Histogram of marks distribution → right-skewed. Correlation: attendance and marks = 0.72 (strong positive). |
| **5. Feature Engineering** | Create `Engagement_Score` = (LMS logins × 0.3 + assignments submitted × 0.7). |
| **6. Model Building** | Train Logistic Regression (baseline) and Random Forest. |
| **7. Evaluation** | Random Forest: Accuracy=88%, Recall=91%, F1=0.87. Better than Logistic (82%). |
| **8. Deployment** | Dashboard for faculty showing at-risk students list. Alerts sent when attendance drops below 60%. |

---

### Q7. Write Python code to create a complete visualization dashboard with Matplotlib (subplot with 4 charts).

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Sample data
departments = ['CSE', 'ECE', 'ME', 'CE', 'EE']
avg_salary = [72000, 65000, 58000, 55000, 60000]
ages = np.random.normal(30, 5, 200)
experience = np.random.uniform(1, 15, 50)
salary = experience * 5000 + np.random.normal(0, 5000, 50) + 30000
ratings = np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Employee Analytics Dashboard', fontsize=16, fontweight='bold')

# 1. Bar Chart
axes[0, 0].bar(departments, avg_salary, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
axes[0, 0].set_title('Average Salary by Department')
axes[0, 0].set_ylabel('Salary (₹)')

# 2. Histogram
axes[0, 1].hist(ages, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Age Distribution')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')

# 3. Scatter Plot
axes[1, 0].scatter(experience, salary, alpha=0.6, color='coral', edgecolors='black')
axes[1, 0].set_title('Experience vs Salary')
axes[1, 0].set_xlabel('Experience (Years)')
axes[1, 0].set_ylabel('Salary (₹)')

# 4. Pie Chart
unique, counts = np.unique(ratings, return_counts=True)
axes[1, 1].pie(counts, labels=unique, autopct='%1.1f%%', colors=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
axes[1, 1].set_title('Rating Distribution')

plt.tight_layout()
plt.savefig('dashboard.png', dpi=150)
plt.show()
```

---

### Q8. Explain the K-Means algorithm and compute two iterations.

**K-Means:** Unsupervised clustering that partitions data into K groups.

**Given:** 2D points: A(1,1), B(2,1), C(4,3), D(5,4), K=2
**Initial centroids:** C1=(1,1), C2=(5,4)

**Iteration 1:**

| Point | d to C1(1,1) | d to C2(5,4) | Cluster |
| :---: | :---: | :---: | :---: |
| A(1,1) | √0 = 0 | √(16+9) = 5 | C1 |
| B(2,1) | √(1+0) = 1 | √(9+9) = 4.24 | C1 |
| C(4,3) | √(9+4) = 3.61 | √(1+1) = 1.41 | C2 |
| D(5,4) | √(16+9) = 5 | √0 = 0 | C2 |

New centroids:
- C1 = ((1+2)/2, (1+1)/2) = **(1.5, 1.0)**
- C2 = ((4+5)/2, (3+4)/2) = **(4.5, 3.5)**

**Iteration 2 (C1=1.5,1.0 and C2=4.5,3.5):**

| Point | d to C1(1.5,1) | d to C2(4.5,3.5) | Cluster |
| :---: | :---: | :---: | :---: |
| A(1,1) | √(0.25) = 0.5 | √(12.25+6.25) = 4.3 | C1 |
| B(2,1) | √(0.25) = 0.5 | √(6.25+6.25) = 3.54 | C1 |
| C(4,3) | √(6.25+4) = 3.2 | √(0.25+0.25) = 0.71 | C2 |
| D(5,4) | √(12.25+9) = 4.61 | √(0.25+0.25) = 0.71 | C2 |

Same clusters → **Converged!**
Final: Cluster1 = {A, B}, Cluster2 = {C, D}

---

### Q9. Explain Data Preprocessing and its steps with code examples for each.

**Data Preprocessing** transforms raw data into a clean format for analysis.

**Step 1: Data Cleaning**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
    'Age': [22, np.nan, 22, 25, 28],
    'Salary': [50000, 60000, 50000, np.nan, 45000]
})

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Name'] = df['Name'].fillna('Unknown')
```

**Step 2: Data Transformation**
```python
# Min-Max Normalization
df['Salary_Normalized'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
```

**Step 3: Data Integration**
```python
df2 = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Department': ['CSE', 'ECE']})
merged = pd.merge(df, df2, on='Name', how='left')
```

**Step 4: Data Reduction**
```python
# Feature selection — drop less important columns
df_reduced = df.drop(columns=['Salary_Normalized'])

# Sampling
df_sample = df.sample(frac=0.5, random_state=42)
```

---

### Q10. Compare Histograms, Boxplots, and Scatterplots. When to use each?

| Feature | Histogram | Boxplot | Scatterplot |
| :--- | :--- | :--- | :--- |
| **Shows** | Frequency distribution | Summary stats (Q1, Q2, Q3, IQR, outliers) | Relationship between 2 variables |
| **Variables** | 1 numeric | 1 numeric (optionally grouped) | 2 numeric |
| **Best For** | Understanding data shape (normal, skewed) | Comparing distributions, detecting outliers | Finding correlations |
| **Outliers** | Not directly visible | Clearly shown as dots | Visible as isolated points |
| **Library** | `plt.hist()` | `sns.boxplot()` | `plt.scatter()` |

**When to Use:**
- **"What does the salary distribution look like?"** → Histogram
- **"Are there outliers in marks across departments?"** → Boxplot
- **"Is there a correlation between study hours and grades?"** → Scatterplot

---

### Q11. Explain Ethical Considerations in Data Science with industry examples.

**1. Data Privacy & Consent:**
- Users must know what data is collected and how it's used.
- **GDPR** mandates explicit opt-in consent.
- **Example:** Aadhaar data breach (India, 2018) — personal details of millions exposed through unsecured APIs.

**2. Algorithmic Bias:**
- Models trained on biased data produce discriminatory results.
- **Example:** Amazon's AI hiring tool (2018) was biased against women because training data came from 10 years of male-dominated resumes.
- **Fix:** Use balanced datasets, audit models for bias, apply fairness constraints.

**3. Transparency & Explainability:**
- Black-box models (deep learning) make decisions humans can't understand.
- **Example:** A loan rejection by an AI system without explanation violates right to fair treatment.
- Tools: LIME (Local Interpretable Model-agnostic Explanations), SHAP values.

**4. Data Security:**
- **Example:** Equifax breach (2017) — 147 million people's Social Security numbers exposed.
- Requires encryption, access controls, regular audits.

**5. Responsible AI:**
- Consider societal impact of AI applications.
- **Example:** Deepfakes used to create fake videos of politicians — raises serious ethical concerns about misinformation.

---

## SECTION C — Long Answer (20 Marks Each, Do Any 2 of 3 = 40 Marks)

---

### Q12. (a) Describe Descriptive and Predictive Data Mining with real-world examples (10 Marks). (b) Explain the complete Data Science Lifecycle with a diagram (10 Marks).

**(a) Descriptive vs Predictive Data Mining:**

**Descriptive Data Mining** — Summarizes data to find patterns.
- **Clustering:** Segment customers into groups (budget, premium, luxury).
- **Association:** "Customers who buy pizza also buy soft drinks" (support=0.65, confidence=0.80).
- **Summarization:** "Average order value increased 15% in Q3."

**Predictive Data Mining** — Uses patterns to forecast.
- **Classification:** Predict if email is spam/not spam.
- **Regression:** Predict house price based on area, location.
- **Time Series:** Predict next month's stock price.

**E-commerce Case Study (Flipkart):**
- **Descriptive:** "60% of electronics buyers are aged 18-30." "Peak sales happen during Diwali (3x normal)."
- **Predictive:** "User X will likely buy a phone case based on recent phone purchase." "Demand for winter clothing will increase 200% in November."

**(b) Data Science Lifecycle:**

```
  ┌──────────────┐
  │  1. Business │
  │  Understanding│
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  2. Data     │
  │  Collection  │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  3. Data     │
  │  Preparation │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  4. EDA &    │
  │  Analysis    │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  5. Modelling│
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 6. Evaluation│
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 7. Deployment│
  │ & Monitoring │
  └──────────────┘
```

Each phase feeds back into previous phases when issues arise — it's iterative, not linear.

---

### Q13. A company's HR dataset has: `Emp_ID, Name, Age, Gender, Department, Years_Experience, Monthly_Salary, Performance_Rating, Joining_Date, City`. Write a complete Python program.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. Load Data ───
df = pd.read_csv('hr_data.csv')
print("Shape:", df.shape)
print("Dtypes:\n", df.dtypes)
print("First 5:\n", df.head())

# ─── 2. Descriptive Statistics ───
print(f"\nMean Salary: ₹{df['Monthly_Salary'].mean():,.2f}")
print(f"Median Experience: {df['Years_Experience'].median()}")
print(f"Mode Gender: {df['Gender'].mode()[0]}")
print(f"Range of Salary: {df['Monthly_Salary'].max() - df['Monthly_Salary'].min()}")
print(f"Variance of Rating: {df['Performance_Rating'].var():.3f}")
print(f"Std Dev of Salary: {df['Monthly_Salary'].std():,.2f}")

# ─── 3. Missing Values ───
print("\nMissing:\n", df.isnull().sum())
df['Monthly_Salary'] = df['Monthly_Salary'].fillna(df['Monthly_Salary'].median())
df['Performance_Rating'] = df['Performance_Rating'].fillna(df['Performance_Rating'].mean())
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])

# ─── 4. Feature Engineering ───
def exp_category(y):
    if y < 3: return 'Junior'
    elif y <= 7: return 'Mid-Level'
    else: return 'Senior'

df['Exp_Level'] = df['Years_Experience'].apply(exp_category)
print("\nExperience Level Distribution:")
print(df['Exp_Level'].value_counts())

# ─── 5. Grouping ───
dept_stats = df.groupby('Department').agg({
    'Monthly_Salary': ['mean', 'median', 'std'],
    'Performance_Rating': 'mean'
}).round(2)
print("\nDepartment Stats:\n", dept_stats)

# ─── 6. Boxplot ───
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Monthly_Salary', data=df, palette='coolwarm')
plt.title('Salary by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ─── 7. Histogram + KDE ───
plt.figure(figsize=(8, 5))
sns.histplot(df['Performance_Rating'], bins=15, kde=True, color='teal')
plt.title('Performance Rating Distribution')
plt.xlabel('Rating')
plt.show()

# ─── 8. Scatter ───
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Years_Experience', y='Monthly_Salary', hue='Department', data=df, alpha=0.7)
plt.title('Experience vs Salary')
plt.show()

# ─── 9. IQR Outliers ───
Q1 = df['Monthly_Salary'].quantile(0.25)
Q3 = df['Monthly_Salary'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df['Monthly_Salary'] < lower) | (df['Monthly_Salary'] > upper)]
print(f"\nIQR: {IQR:,.0f}, Bounds: [{lower:,.0f}, {upper:,.0f}]")
print(f"Outliers: {len(outliers)}")

# ─── 10. Save ───
df.to_csv('hr_cleaned.csv', index=False)
print("Saved!")
```

---

### Q14. (a) Explain all types of Machine Learning with examples (10 Marks). (b) Describe k-NN and K-Means with manual examples (10 Marks).

**(a) Types of Machine Learning:**

**1. Supervised Learning:**
- Labelled data → learns input-output mapping.
- Classification: Spam detection, disease diagnosis.
- Regression: House price prediction, sales forecasting.

**2. Unsupervised Learning:**
- Unlabelled data → discovers hidden structure.
- Clustering: Customer segmentation.
- Association: Market basket analysis.
- Dimensionality Reduction: PCA.

**3. Semi-Supervised Learning:**
- Small labelled + large unlabelled data.
- Example: Google Photos — a few labelled face images help identify people across thousands of unlabelled photos.

**4. Reinforcement Learning:**
- Agent learns through rewards/penalties in an environment.
- Example: Self-driving cars — agent gets positive reward for staying in lane, negative for crossing.

**(b) k-NN Manual Example:**

Training data:

| Point | X | Y | Class |
|:---:|:---:|:---:|:---|
| A | 2 | 3 | Cat |
| B | 3 | 4 | Cat |
| C | 6 | 7 | Dog |
| D | 7 | 8 | Dog |

New point P(4, 5), k=3:
- d(P,A) = √(4+4) = 2.83
- d(P,B) = √(1+1) = 1.41
- d(P,C) = √(4+4) = 2.83
- d(P,D) = √(9+9) = 4.24

3 nearest: B(Cat,1.41), A(Cat,2.83), C(Dog,2.83) → **2 Cat, 1 Dog → Cat**

**(c) K-Means Manual Example:**

Data: {2, 3, 8, 9, 10}, K=2, C1=2, C2=10

| Point | d to C1=2 | d to C2=10 | Cluster |
|:---:|:---:|:---:|:---:|
| 2 | 0 | 8 | C1 |
| 3 | 1 | 7 | C1 |
| 8 | 6 | 2 | C2 |
| 9 | 7 | 1 | C2 |
| 10 | 8 | 0 | C2 |

New: C1=(2+3)/2=**2.5**, C2=(8+9+10)/3=**9.0**

Iteration 2 with C1=2.5, C2=9.0 → same assignments → **Converged!**

---

**— End of Paper 5 (Medium) —**
