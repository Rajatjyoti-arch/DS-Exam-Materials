# Introduction to Data Science — Final Exam Paper 2 (Easy)

**Course Code:** BCSE2C05 | **Semester:** II | **Max Marks:** 100 | **Time:** 3 Hours
**Difficulty:** ★☆☆☆☆ Easy

---

## SECTION A — Short Answer (5 Marks Each, Do Any 4 of 5 = 20 Marks)

---

### Q1. What is Data Science? List any four applications of Data Science.

**Data Science** is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data.

**Four Applications:**
1. **Healthcare** — Predicting disease outbreaks, drug discovery, patient diagnosis.
2. **E-Commerce** — Product recommendation systems (Amazon, Flipkart).
3. **Finance** — Fraud detection in credit card transactions.
4. **Social Media** — Sentiment analysis, targeted advertising (Facebook, Instagram).

---

### Q2. Differentiate between Discrete and Continuous data with examples.

| Aspect | Discrete Data | Continuous Data |
| :--- | :--- | :--- |
| **Definition** | Countable, finite values. | Measurable, can take any value within a range. |
| **Values** | Whole numbers only. | Decimal/fractional values possible. |
| **Example** | Number of students in a class (30, 31, 32) | Height of students (5.5 ft, 5.7 ft, 6.1 ft) |
| **Visualization** | Bar chart | Histogram |

---

### Q3. Write Python code to create a NumPy array and find its mean, sum, and maximum value.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

print("Mean:", np.mean(arr))      # Output: 30.0
print("Sum:", np.sum(arr))        # Output: 150
print("Max:", np.max(arr))        # Output: 50
print("Min:", np.min(arr))        # Output: 10
print("Std Dev:", np.std(arr))    # Output: 14.14
```

---

### Q4. What is the difference between Mean, Median, and Mode? Give an example.

| Measure | Definition | Formula/Method | Example (Data: 2, 3, 3, 5, 7) |
| :--- | :--- | :--- | :--- |
| **Mean** | Average of all values | Sum / Count | (2+3+3+5+7)/5 = **4.0** |
| **Median** | Middle value when sorted | Middle position | Sorted: 2,3,**3**,5,7 → **3** |
| **Mode** | Most frequently occurring value | Highest frequency | **3** (appears twice) |

---

### Q5. What is Matplotlib? Write a code to plot a simple line chart.

**Matplotlib** is a Python library used for creating static, animated, and interactive visualizations. It is the most widely used plotting library in Python for data science.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]

plt.plot(x, y, marker='o', color='blue', linestyle='-')
plt.title('Simple Line Chart')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.grid(True)
plt.show()
```

---

## SECTION B — Analytical Questions (8 Marks Each, Do Any 5 of 6 = 40 Marks)

---

### Q6. Explain the Data Science Lifecycle with a neat diagram.

**Data Science Lifecycle** is a systematic approach to solving problems using data:

```
Problem Definition → Data Collection → Data Cleaning → EDA → 
Model Building → Evaluation → Deployment
```

**Phases:**

1. **Problem Definition** — Identify the business question. *Example:* "Which customers will churn?"
2. **Data Collection** — Gather data from databases, APIs, surveys. *Example:* Customer transaction logs.
3. **Data Cleaning** — Handle missing values, duplicates, inconsistencies. *Example:* Fill missing ages with median.
4. **EDA (Exploratory Data Analysis)** — Visualize data using plots, compute statistics. *Example:* Histogram of purchase amounts.
5. **Model Building** — Train ML algorithms. *Example:* Logistic regression for churn prediction.
6. **Evaluation** — Test model using accuracy, F1-score. *Example:* 92% accuracy on test data.
7. **Deployment** — Put model into production. *Example:* REST API for real-time predictions.

---

### Q7. What is a DataFrame in Pandas? Write code to create a DataFrame and perform basic operations.

A **DataFrame** is a 2D, size-mutable, tabular data structure in Pandas with labelled rows and columns (like a spreadsheet or SQL table).

```python
import pandas as pd

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [22, 25, 23, 28],
    'Marks': [85, 90, 78, 92]
}
df = pd.DataFrame(data)

# Basic operations
print(df.head())          # First 5 rows
print(df.shape)           # (4, 3)
print(df.describe())      # Statistical summary
print(df['Age'].mean())   # Mean age = 24.5
print(df.dtypes)          # Data types of each column

# Filtering
top_students = df[df['Marks'] > 80]
print(top_students)

# Sorting
df_sorted = df.sort_values('Marks', ascending=False)
print(df_sorted)
```

---

### Q8. What are the different types of data in Data Science? Explain with examples.

**1. Based on Structure:**

| Type | Description | Example |
| :--- | :--- | :--- |
| **Structured** | Organized in rows & columns. | SQL databases, CSV files, spreadsheets |
| **Unstructured** | No predefined format. | Emails, images, videos, social media posts |
| **Semi-Structured** | Partially organized with tags/markers. | JSON, XML, HTML files |

**2. Based on Attribute Types:**

| Attribute | Description | Example |
| :--- | :--- | :--- |
| **Nominal** | Categories with no order. | Gender (Male/Female), Color (Red/Blue) |
| **Ordinal** | Categories with meaningful order. | Rating (Poor < Average < Good < Excellent) |
| **Binary** | Only two values. | Pass/Fail, Yes/No |
| **Numeric (Discrete)** | Countable whole numbers. | Number of children: 0, 1, 2, 3 |
| **Numeric (Continuous)** | Measurable, any value in a range. | Temperature: 36.5°C, Weight: 65.2 kg |

---

### Q9. What is Data Preprocessing? Why is it needed?

**Data Preprocessing** is the process of transforming raw data into a clean, usable format before analysis or model building.

**Why it is needed:**
- Raw data is often incomplete, inconsistent, noisy, and full of errors.
- ML models cannot work with missing or poorly formatted data.
- Clean data leads to better model accuracy and reliable insights.

**Steps of Data Preprocessing:**

| Step | Description | Example |
| :--- | :--- | :--- |
| **Data Cleaning** | Handle missing values, noise, duplicates. | Fill NaN with mean, remove duplicates. |
| **Data Integration** | Combine data from multiple sources. | Merging customer data from CRM and billing. |
| **Data Transformation** | Normalize, aggregate, or encode data. | Min-Max normalization, one-hot encoding. |
| **Data Reduction** | Reduce volume while preserving information. | PCA, feature selection, sampling. |

---

### Q10. Explain the different sources of data with examples.

| Source | Description | Example |
| :--- | :--- | :--- |
| **Time Series Data** | Data collected at regular intervals over time. | Daily stock prices, monthly rainfall data. |
| **Transactional Data** | Records of business transactions. | E-commerce purchase records, bank transactions. |
| **Biological Data** | Data from living organisms and medical studies. | DNA sequences, patient health records. |
| **Spatial Data** | Data associated with geographic locations. | GPS coordinates, satellite images, maps. |
| **Social Network Data** | Data from social media interactions. | Twitter tweets, Facebook likes, Instagram comments. |

**Data Evolution:** Over time, data formats, volume, and sources evolve — new sources like IoT sensors and wearables generate continuous streams of data requiring real-time processing.

---

### Q11. Write a Python program to handle missing values using Pandas.

```python
import pandas as pd
import numpy as np

# Create DataFrame with missing values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [22, np.nan, 25, np.nan, 28],
    'Salary': [50000, 60000, np.nan, 45000, np.nan]
}
df = pd.DataFrame(data)

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Method 1: Fill with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
print("\nAfter filling Age with mean:\n", df)

# Method 2: Fill with median
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
print("\nAfter filling Salary with median:\n", df)

# Method 3: Drop rows with any NaN (on original data)
df_dropped = df.dropna()
print("\nAfter dropping rows with NaN:\n", df_dropped)

# Method 4: Forward fill
df_ffill = df.ffill()
print("\nAfter forward fill:\n", df_ffill)

# Method 5: Backward fill
df_bfill = df.bfill()
print("\nAfter backward fill:\n", df_bfill)
```

---

## SECTION C — Long Answer (20 Marks Each, Do Any 2 of 3 = 40 Marks)

---

### Q12. Explain Big Data and its characteristics. How does Big Data form the foundation of Data Science?

**(a) Big Data (10 Marks):**

**Big Data** refers to extremely large and complex datasets that cannot be processed using traditional data management tools.

**5 V's of Big Data:**

| V | Description | Example |
| :--- | :--- | :--- |
| **Volume** | Huge amount of data. | Facebook generates 4+ petabytes daily. |
| **Velocity** | Speed of data generation. | Stock market tickers updating every millisecond. |
| **Variety** | Different data types. | Text, images, audio, video, sensor data. |
| **Veracity** | Data quality and accuracy. | Social media may have fake news/bots. |
| **Value** | Usefulness of processed data. | Customer insights leading to better marketing. |

**Impact on Industries:**
1. **Healthcare** — Predictive diagnosis using patient records.
2. **Retail** — Amazon's recommendation engine.
3. **Banking** — Real-time fraud detection.
4. **Transportation** — Uber's surge pricing and route optimization.
5. **Education** — Personalized learning platforms.

**(b) Big Data as Foundation of Data Science (10 Marks):**

Data Science relies on Big Data because:
- **Data Collection** — Big Data provides the raw material (logs, sensors, social media).
- **Storage** — Technologies like Hadoop and Spark enable storing and processing massive datasets.
- **Analysis** — Statistical and ML techniques extract patterns from large datasets.
- **Visualization** — Tools like Tableau and Matplotlib present findings.
- **Decision Making** — Insights from Big Data drive business strategy.

**Case Study — Netflix:**
Netflix collects viewing data from 200M+ users, analyses it using ML models to recommend content, and even decides which original shows to produce based on data-driven insights.

---

### Q13. Write a Python program to read a CSV file, perform data analysis using Pandas, and create visualizations.

**Scenario:** A CSV file `students.csv` has columns: `Roll_No, Name, Age, Department, CGPA, Attendance`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Read the CSV file
df = pd.read_csv('students.csv')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# Task 2: Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())
print(f"\nMean CGPA: {df['CGPA'].mean():.2f}")
print(f"Median Age: {df['Age'].median()}")
print(f"Mode of Department: {df['Department'].mode()[0]}")

# Task 3: Handle missing values
print("\nMissing values:")
print(df.isnull().sum())
df['CGPA'] = df['CGPA'].fillna(df['CGPA'].mean())
df['Attendance'] = df['Attendance'].fillna(df['Attendance'].median())

# Task 4: Filter students with CGPA > 8
toppers = df[df['CGPA'] > 8.0]
print(f"\nStudents with CGPA > 8: {len(toppers)}")
print(toppers[['Name', 'CGPA']])

# Task 5: Group by Department
dept_avg = df.groupby('Department')['CGPA'].mean()
print("\nAverage CGPA by Department:")
print(dept_avg)

# Task 6: Histogram of CGPA
plt.figure(figsize=(8, 5))
plt.hist(df['CGPA'], bins=10, color='steelblue', edgecolor='black')
plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Frequency')
plt.show()

# Task 7: Bar chart of department-wise average CGPA
plt.figure(figsize=(8, 5))
dept_avg.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Average CGPA by Department')
plt.xlabel('Department')
plt.ylabel('Average CGPA')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 8: Save cleaned data
df.to_csv('students_cleaned.csv', index=False)
print("Cleaned data saved!")
```

---

### Q14. Explain Supervised and Unsupervised Machine Learning. Describe k-NN and K-Means with examples.

**(a) Supervised vs Unsupervised Learning (8 Marks):**

| Aspect | Supervised Learning | Unsupervised Learning |
| :--- | :--- | :--- |
| **Data** | Labelled (input + output known) | Unlabelled (only input) |
| **Goal** | Predict output for new inputs | Discover hidden patterns |
| **Types** | Classification, Regression | Clustering, Association |
| **Example** | Spam detection (spam/not spam) | Customer segmentation |
| **Algorithms** | k-NN, Decision Tree, SVM | K-Means, DBSCAN, Apriori |

**(b) k-Nearest Neighbours (k-NN) (6 Marks):**

k-NN is a supervised algorithm that classifies a new point based on the majority class of its k nearest neighbours.

**Steps:**
1. Choose k (e.g., k = 3).
2. Calculate distance from new point to all training points (Euclidean distance).
3. Pick the k closest points.
4. Assign the majority class.

**Example:** Classifying a fruit as Apple or Orange based on weight and colour:
- New fruit: Weight=150g → 3 nearest neighbours: 2 Apples, 1 Orange → **Apple**

**(c) K-Means Clustering (6 Marks):**

K-Means is an unsupervised algorithm that partitions data into K clusters.

**Steps:**
1. Choose K (number of clusters).
2. Initialize K centroids randomly.
3. Assign each point to the nearest centroid.
4. Recompute centroids as the mean of each cluster.
5. Repeat until convergence.

**Example:** Given points {1, 2, 8, 9, 10}, K=2, initial centroids C1=1, C2=8:
- **Iteration 1:** Cluster1={1,2}, Cluster2={8,9,10} → C1=1.5, C2=9.0
- **Iteration 2:** Cluster1={1,2}, Cluster2={8,9,10} → C1=1.5, C2=9.0 (converged ✓)

---

**— End of Paper 2 (Easy) —**
