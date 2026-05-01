# Introduction to Data Science — Final Exam Paper 4 (Medium)

**Course Code:** BCSE2C05 | **Semester:** II | **Max Marks:** 100 | **Time:** 3 Hours
**Difficulty:** ★★★☆☆ Medium

---

## SECTION A — Short Answer (5 Marks Each, Do Any 4 of 5 = 20 Marks)

---

### Q1. Explain Variance and Standard Deviation with a numerical example.

**Variance (σ²)** measures how spread out data points are from the mean. **Standard Deviation (σ)** is the square root of variance — expressed in the same units as data.

**Example:** Data = {4, 6, 8, 10, 12}

**Step 1:** Mean = (4+6+8+10+12)/5 = 40/5 = **8**

**Step 2:** Deviations from mean:

| xᵢ | xᵢ − μ | (xᵢ − μ)² |
| :---: | :---: | :---: |
| 4 | −4 | 16 |
| 6 | −2 | 4 |
| 8 | 0 | 0 |
| 10 | 2 | 4 |
| 12 | 4 | 16 |

**Step 3:** Variance = Σ(xᵢ − μ)² / N = (16+4+0+4+16)/5 = 40/5 = **8.0**

**Step 4:** Standard Deviation = √8 = **2.83**

---

### Q2. Differentiate between Data Integration and Data Transformation.

| Aspect | Data Integration | Data Transformation |
| :--- | :--- | :--- |
| **Definition** | Combining data from multiple sources into a single unified dataset. | Converting data from one format/structure to another. |
| **Purpose** | Create a consistent, comprehensive dataset. | Make data suitable for analysis or modelling. |
| **Challenges** | Schema conflicts, naming inconsistencies, duplicate detection. | Choosing right scaling method, handling categorical data. |
| **Techniques** | Entity matching, schema alignment, ETL pipelines. | Normalization, encoding, aggregation, discretization. |
| **Example** | Merging hospital patient records from 3 different databases. | Converting salary from ₹ to normalized [0,1] range. |

---

### Q3. Write Python code to create a scatter plot and add a title, labels, and grid.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
x = np.random.rand(50) * 100   # Study hours
y = x * 0.6 + np.random.randn(50) * 10 + 20  # Marks

# Create scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='darkblue', marker='o', alpha=0.7, edgecolors='black')
plt.title('Study Hours vs Exam Marks', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Marks Obtained', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

---

### Q4. What is the difference between `loc` and `iloc` in Pandas? Give examples.

| Feature | `loc` | `iloc` |
| :--- | :--- | :--- |
| **Indexing** | **Label-based** (uses row/column names). | **Integer position-based** (uses index numbers). |
| **Syntax** | `df.loc[row_label, col_label]` | `df.iloc[row_index, col_index]` |
| **Inclusive** | Both start and end are inclusive. | End is exclusive (like Python slicing). |

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [22, 25, 23, 28],
    'Marks': [85, 90, 78, 92]
}, index=['a', 'b', 'c', 'd'])

# loc — label-based
print(df.loc['a', 'Name'])       # Output: Alice
print(df.loc['a':'c', 'Name'])   # Output: Alice, Bob, Charlie (inclusive)

# iloc — position-based
print(df.iloc[0, 0])             # Output: Alice
print(df.iloc[0:3, 0])           # Output: Alice, Bob, Charlie (end exclusive)
```

---

### Q5. What is Folium? Write code to display a map with a marker.

**Folium** is a Python library for creating interactive maps. It uses the Leaflet.js library and renders maps as HTML.

```python
import folium

# Create a map centered on New Delhi
m = folium.Map(location=[28.6139, 77.2090], zoom_start=12)

# Add a marker
folium.Marker(
    location=[28.6139, 77.2090],
    popup='New Delhi, India',
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

# Save as HTML
m.save('delhi_map.html')
print("Map saved as delhi_map.html")
```

---

## SECTION B — Analytical Questions (8 Marks Each, Do Any 5 of 6 = 40 Marks)

---

### Q6. Explain the k-NN algorithm. How does the value of k affect model performance? Compute a classification manually.

**k-NN (k-Nearest Neighbours):** A supervised, lazy learning algorithm that classifies a new data point by looking at the majority class among its k nearest neighbours.

**Effect of k:**
- **k too small (e.g., 1):** Overfitting — sensitive to noise.
- **k too large (e.g., n):** Underfitting — model becomes too generalized.
- **Best practice:** Use odd k to avoid ties; find optimal k via cross-validation.

**Manual Example:**

Training data:

| Point | X | Y | Class |
| :---: | :---: | :---: | :--- |
| A | 1 | 2 | Red |
| B | 2 | 3 | Red |
| C | 5 | 5 | Blue |
| D | 6 | 7 | Blue |
| E | 3 | 3 | Red |

**New point P = (4, 4), k = 3:**

Euclidean distances:
- d(P, A) = √((4-1)² + (4-2)²) = √(9+4) = √13 = **3.61**
- d(P, B) = √((4-2)² + (4-3)²) = √(4+1) = √5 = **2.24**
- d(P, C) = √((4-5)² + (4-5)²) = √(1+1) = √2 = **1.41**
- d(P, D) = √((4-6)² + (4-7)²) = √(4+9) = √13 = **3.61**
- d(P, E) = √((4-3)² + (4-3)²) = √(1+1) = √2 = **1.41**

3 nearest: C (Blue, 1.41), E (Red, 1.41), B (Red, 2.24)

**Majority vote: 2 Red, 1 Blue → Predicted: Red**

---

### Q7. Given a dataset, write Pandas code to compute central tendency, handle missing data, and group by a column.

```python
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'Department': ['CSE', 'ECE', 'CSE', 'ME', 'ECE', 'CSE', 'ME', 'ECE'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Salary': [55000, 60000, np.nan, 45000, np.nan, 70000, 48000, 52000],
    'Rating': [4.2, np.nan, 3.8, 4.5, 3.9, np.nan, 4.1, 4.3]
}
df = pd.DataFrame(data)

# Central tendency
print("Mean Salary:", df['Salary'].mean())
print("Median Salary:", df['Salary'].median())
print("Mode Department:", df['Department'].mode()[0])

# Missing values
print("\nMissing values:\n", df.isnull().sum())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
print("\nAfter imputation:\n", df.isnull().sum())

# Group by Department
dept_summary = df.groupby('Department').agg({
    'Salary': ['mean', 'max', 'min'],
    'Rating': 'mean'
}).round(2)
print("\nDepartment-wise Summary:")
print(dept_summary)
```

---

### Q8. Explain Data Reduction. Compare PCA with Feature Selection.

**Data Reduction** reduces the volume of data while preserving key information for efficient analysis.

**Why needed:**
- Reduces computation time and memory.
- Avoids the curse of dimensionality.
- Improves model performance by removing redundant features.

| Aspect | PCA (Principal Component Analysis) | Feature Selection |
| :--- | :--- | :--- |
| **Approach** | Creates **new** features (principal components) as linear combinations of original features. | Selects a **subset** of existing features. |
| **Original Features** | Transformed into new axes; originals are lost. | Original features are retained. |
| **Interpretability** | Low — components are mathematical constructs. | High — selected features are meaningful. |
| **Information Loss** | Minimal if enough components retained. | Possible if important features are dropped. |
| **Techniques** | Eigenvalue decomposition, SVD | Correlation analysis, Chi-square, Mutual Information |
| **Use Case** | High-dimensional data (images, genetics) | Tabular data with known feature meanings |

**Other Data Reduction Techniques:**
- **Numerosity Reduction** — Represent data with models (regression coefficients) instead of raw values.
- **Data Compression** — Lossless (ZIP) or lossy (JPEG) encoding.
- **Sampling** — Random, systematic, or stratified sampling to reduce dataset size.

---

### Q9. Write Python code to create a Boxplot and a Histogram using Seaborn and Matplotlib.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample data
np.random.seed(42)
data = {
    'Department': np.random.choice(['CSE', 'ECE', 'ME'], 100),
    'Salary': np.random.normal(50000, 15000, 100),
    'Performance': np.random.normal(75, 10, 100)
}
df = pd.DataFrame(data)

# ── Boxplot (Seaborn) ──
plt.figure(figsize=(10, 5))
sns.boxplot(x='Department', y='Salary', data=df, palette='pastel')
plt.title('Salary Distribution by Department', fontsize=14)
plt.xlabel('Department')
plt.ylabel('Salary (₹)')
plt.show()

# ── Histogram (Matplotlib) ──
plt.figure(figsize=(8, 5))
plt.hist(df['Performance'], bins=15, color='coral', edgecolor='black', alpha=0.7)
plt.title('Distribution of Performance Scores', fontsize=14)
plt.xlabel('Performance Score')
plt.ylabel('Frequency')
plt.axvline(df['Performance'].mean(), color='red', linestyle='--',
            label=f'Mean = {df["Performance"].mean():.1f}')
plt.legend()
plt.show()
```

**Interpretation:**
- **Boxplot** shows median (center line), Q1-Q3 (box), whiskers (1.5×IQR), and outliers (dots beyond whiskers).
- **Histogram** shows frequency distribution. The mean line helps identify skewness.

---

### Q10. Explain the K-Means Clustering algorithm. Perform one iteration with given data.

**K-Means** is an unsupervised partitioning algorithm that groups data into K clusters.

**Algorithm:**
1. Choose K → Initialize K centroids
2. **Assign** each point to nearest centroid
3. **Update** centroids = mean of assigned points
4. **Repeat** until convergence

**Given:** Points = {1, 3, 7, 8, 9}, K = 2, C1 = 1, C2 = 7

**Iteration 1 — Assignment:**

| Point | &#124;d to C1=1&#124; | &#124;d to C2=7&#124; | Cluster |
| :---: | :---: | :---: | :---: |
| 1 | 0 | 6 | C1 |
| 3 | 2 | 4 | C1 |
| 7 | 6 | 0 | C2 |
| 8 | 7 | 1 | C2 |
| 9 | 8 | 2 | C2 |

**New centroids:**
- C1 = (1+3)/2 = **2.0**
- C2 = (7+8+9)/3 = **8.0**

**Iteration 2 — Assignment (C1=2.0, C2=8.0):**

| Point | &#124;d to C1=2&#124; | &#124;d to C2=8&#124; | Cluster |
| :---: | :---: | :---: | :---: |
| 1 | 1 | 7 | C1 |
| 3 | 1 | 5 | C1 |
| 7 | 5 | 1 | C2 |
| 8 | 6 | 0 | C2 |
| 9 | 7 | 1 | C2 |

Same clusters → **Converged!**
Final: Cluster1 = {1, 3}, Cluster2 = {7, 8, 9}

---

### Q11. What is Ethical Data Science? Discuss privacy, bias, and transparency with examples.

**Ethical Data Science** ensures that data collection, analysis, and deployment of models are conducted fairly, transparently, and without causing harm.

**1. Data Privacy:**
- Users must consent to data collection.
- Data must be stored securely (encryption, access controls).
- Regulations: GDPR (EU), IT Act (India), CCPA (California).
- **Example:** WhatsApp faced backlash in India (2021) for updating privacy policy to share data with Facebook without clear user consent.

**2. Algorithmic Bias:**
- Models can discriminate based on race, gender, or socioeconomic status if trained on biased data.
- **Example:** COMPAS algorithm (used in US courts) predicted Black defendants as higher recidivism risk compared to White defendants with similar profiles.
- **Mitigation:** Balanced training data, fairness-aware algorithms, regular audits.

**3. Transparency:**
- Users should understand how decisions affecting them are made.
- "Black box" models (deep neural networks) are hard to interpret.
- **Explainable AI (XAI)** techniques: LIME, SHAP values.
- **Example:** EU's GDPR gives citizens the "right to explanation" for automated decisions.

**4. Informed Consent:**
- **Example:** Cambridge Analytica (2018) — harvested 87M Facebook profiles without proper consent for political targeting.

---

## SECTION C — Long Answer (20 Marks Each, Do Any 2 of 3 = 40 Marks)

---

### Q12. (a) Explain Big Data, its 5 V's, and impact on industries (10 Marks). (b) Describe the Data Science Lifecycle with a real-world case study (10 Marks).

**(a) Big Data & 5 V's:**

**Big Data** = datasets too large and complex for traditional tools.

| V | Meaning | Example |
| :--- | :--- | :--- |
| **Volume** | Terabytes to petabytes of data | YouTube: 500+ hours of video uploaded every minute |
| **Velocity** | Speed of generation & processing | Credit card transactions processed in milliseconds |
| **Variety** | Different data types | Text + images + GPS + audio from a single app |
| **Veracity** | Quality, accuracy, trustworthiness | Twitter data may contain bot-generated fake content |
| **Value** | Business insights after processing | Netflix saves $1B/year through recommendation engine |

**Industry Impact:**
1. **Healthcare** — Real-time patient monitoring via wearables.
2. **Banking** — Real-time fraud detection on millions of transactions.
3. **Retail** — Demand forecasting and inventory management.
4. **Smart Cities** — Traffic optimization using IoT sensor data.
5. **Agriculture** — Satellite + weather data for precision farming.

**(b) Data Science Lifecycle — Zomato Case Study:**

| Phase | Application |
| :--- | :--- |
| **Problem** | "Predict restaurant delivery time to improve customer satisfaction." |
| **Data Collection** | Order history, restaurant location, driver GPS, traffic APIs, weather data. |
| **Data Cleaning** | Remove cancelled orders, fill missing delivery times, normalize addresses. |
| **EDA** | Visualize avg delivery time by city, day of week, cuisine type. Find peak hours. |
| **Feature Engineering** | Distance (restaurant to customer), time of order, restaurant rating, traffic index. |
| **Modelling** | Train Random Forest and XGBoost regression models. |
| **Evaluation** | Compare MAE and RMSE on test data. XGBoost: MAE = 4.2 mins. |
| **Deployment** | Real-time prediction displayed to customers: "Estimated delivery: 32 mins." |

---

### Q13. Write a complete Python program for employee data analysis using Pandas, NumPy, Matplotlib, and Seaborn.

**File:** `sales_data.csv` with columns: `Sales_ID, Salesperson, Region, Product, Units_Sold, Revenue, Quarter, Year`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. Load & Inspect ───
df = pd.read_csv('sales_data.csv')
print("Shape:", df.shape)
print("Data Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())
print("\nStatistical Summary:\n", df.describe())

# ─── 2. Central Tendency ───
print(f"\nMean Revenue: ₹{df['Revenue'].mean():,.2f}")
print(f"Median Revenue: ₹{df['Revenue'].median():,.2f}")
print(f"Mode of Region: {df['Region'].mode()[0]}")
print(f"Std Dev of Units_Sold: {df['Units_Sold'].std():.2f}")

# ─── 3. Missing Values ───
print("\nMissing Values:\n", df.isnull().sum())
df['Revenue'] = df['Revenue'].fillna(df['Revenue'].median())
df['Units_Sold'] = df['Units_Sold'].fillna(df['Units_Sold'].mean())
df['Region'] = df['Region'].fillna(df['Region'].mode()[0])

# ─── 4. Derived Column ───
df['Revenue_Category'] = pd.cut(df['Revenue'],
    bins=[0, 50000, 150000, float('inf')],
    labels=['Low', 'Medium', 'High'])
print("\nRevenue Category Distribution:")
print(df['Revenue_Category'].value_counts())

# ─── 5. Group by Region ───
region_summary = df.groupby('Region').agg({
    'Revenue': ['sum', 'mean'],
    'Units_Sold': 'sum'
}).round(2)
print("\nRegion-wise Summary:")
print(region_summary)

# ─── 6. Boxplot ───
plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='Revenue', data=df, palette='Set2')
plt.title('Revenue Distribution by Region')
plt.ylabel('Revenue (₹)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ─── 7. Histogram ───
plt.figure(figsize=(8, 5))
plt.hist(df['Units_Sold'], bins=20, color='steelblue', edgecolor='black')
plt.title('Distribution of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.show()

# ─── 8. Heatmap (correlation) ───
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# ─── 9. IQR Outlier Detection ───
Q1 = df['Revenue'].quantile(0.25)
Q3 = df['Revenue'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['Revenue'] < lower) | (df['Revenue'] > upper)]
print(f"\nOutlier Bounds: [{lower:,.0f}, {upper:,.0f}]")
print(f"Number of Outliers: {len(outliers)}")

# ─── 10. Save ───
df.to_csv('sales_cleaned.csv', index=False)
print("\nSaved to sales_cleaned.csv")
```

---

### Q14. (a) Compare Descriptive and Predictive Data Mining with case studies (8 Marks). (b) Explain sampling techniques with examples (12 Marks).

**(a) Descriptive vs Predictive Mining:**

| Aspect | Descriptive | Predictive |
| :--- | :--- | :--- |
| **Goal** | Summarize past data | Forecast future outcomes |
| **Question** | "What happened?" | "What will happen?" |
| **Techniques** | Clustering, Association Rules | Classification, Regression |
| **Output** | Patterns, dashboards | Predictions, scores |

**Case Study — Banking:**
- **Descriptive:** "65% of customers who defaulted had credit utilization > 80%." (pattern)
- **Predictive:** "Customer X has 72% probability of defaulting next month." (forecast)

**Case Study — Retail:**
- **Descriptive:** Market basket analysis → "Customers who buy bread also buy butter (support=0.7)."
- **Predictive:** "Customer segment A will spend ₹15,000 next quarter based on trend analysis."

**(b) Sampling Techniques:**

| Technique | Method | Example | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Simple Random** | Every item has equal chance. | Drawing 100 names from a hat. | Unbiased. | May miss subgroups. |
| **Systematic** | Select every k-th item. | Every 10th product on assembly line. | Easy to implement. | Bias if data has cycles. |
| **Stratified** | Divide into strata, sample proportionally. | Sample students from each department proportionally. | All groups represented. | Need to know strata. |
| **Cluster** | Divide into clusters, randomly select entire clusters. | Select 5 random cities, survey all households in them. | Cost-effective for large areas. | High variability between clusters. |
| **Convenience** | Sample whatever is easily available. | Surveying friends on campus. | Quick and cheap. | Highly biased. |

**Numerical Example — Stratified Sampling:**
- Population: 10,000 employees
- Strata: Engineering (4,000), Marketing (3,000), HR (2,000), Finance (1,000)
- Sample size: 500
- Proportional allocation: Eng=200, Mkt=150, HR=100, Fin=50

---

**— End of Paper 4 (Medium) —**
