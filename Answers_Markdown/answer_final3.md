# Introduction to Data Science — Final Exam Paper 3 (Easy)

**Course Code:** BCSE2C05 | **Semester:** II | **Max Marks:** 100 | **Time:** 3 Hours
**Difficulty:** ★☆☆☆☆ Easy

---

## SECTION A — Short Answer (5 Marks Each, Do Any 4 of 5 = 20 Marks)

---

### Q1. What is Data Mining? Differentiate between Descriptive and Predictive Data Mining.

**Data Mining** is the process of discovering patterns, correlations, and useful knowledge from large datasets using statistical and computational techniques.

| Aspect | Descriptive Mining | Predictive Mining |
| :--- | :--- | :--- |
| **Purpose** | Summarizes past data ("what happened"). | Predicts future outcomes ("what will happen"). |
| **Techniques** | Clustering, Association, Summarization | Classification, Regression, Forecasting |
| **Example** | "70% of buyers aged 20-30 prefer online shopping." | "Customer X has 85% chance of buying product Y." |

---

### Q2. What is a Pandas DataFrame? How do you create one from a dictionary?

A **DataFrame** is a 2D, labelled data structure in Pandas — like a spreadsheet with rows and columns.

```python
import pandas as pd

data = {
    'Name': ['Rajat', 'Priya', 'Aman'],
    'Age': [21, 22, 20],
    'CGPA': [8.5, 9.0, 7.8]
}
df = pd.DataFrame(data)
print(df)
```

**Output:**
```
    Name  Age  CGPA
0  Rajat   21   8.5
1  Priya   22   9.0
2   Aman   20   7.8
```

---

### Q3. Explain Range, Quartiles, and Interquartile Range (IQR) with an example.

Given data: `{5, 10, 15, 20, 25, 30, 35}`

| Measure | Definition | Calculation | Result |
| :--- | :--- | :--- | :--- |
| **Range** | Max − Min | 35 − 5 | **30** |
| **Q1** | 25th percentile (median of lower half) | Median of {5, 10, 15} | **10** |
| **Q2** | 50th percentile (median) | Middle value | **20** |
| **Q3** | 75th percentile (median of upper half) | Median of {25, 30, 35} | **30** |
| **IQR** | Q3 − Q1 | 30 − 10 | **20** |

IQR is useful for detecting outliers: values below Q1−1.5×IQR or above Q3+1.5×IQR are outliers.

---

### Q4. What is Seaborn? How is it different from Matplotlib?

**Seaborn** is a Python visualization library built on top of Matplotlib that provides a high-level interface for creating attractive statistical graphics.

| Feature | Matplotlib | Seaborn |
| :--- | :--- | :--- |
| **Level** | Low-level (more control) | High-level (easier syntax) |
| **Default Style** | Basic, plain | Beautiful, modern themes |
| **Statistical Plots** | Manual effort needed | Built-in (boxplot, heatmap, violin) |
| **DataFrames** | Not natively supported | Works directly with Pandas DataFrames |
| **Code Required** | More lines | Fewer lines for same plot |

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Bill Distribution by Day')
plt.show()
```

---

### Q5. List the roles in a Data Science team and briefly explain each.

| Role | Responsibility |
| :--- | :--- |
| **Data Scientist** | Analyses data, builds ML models, extracts insights. |
| **Data Engineer** | Designs and maintains data pipelines and infrastructure. |
| **Data Analyst** | Explores data, creates reports and dashboards. |
| **ML Engineer** | Deploys and scales ML models into production. |
| **Business Analyst** | Translates business needs into data requirements. |
| **Domain Expert** | Provides subject-matter knowledge for proper interpretation. |

---

## SECTION B — Analytical Questions (8 Marks Each, Do Any 5 of 6 = 40 Marks)

---

### Q6. Explain Data Collection Strategies. What are the different ways to collect data?

**Data Collection** is the systematic process of gathering information from relevant sources.

**Strategies:**

| Strategy | Description | Example |
| :--- | :--- | :--- |
| **Surveys/Questionnaires** | Collecting responses from participants. | Google Forms survey on customer satisfaction. |
| **Interviews** | One-on-one structured conversations. | Interviewing doctors for medical research. |
| **Observation** | Watching and recording behaviour. | Tracking user clicks on a website. |
| **Experiments** | Controlled tests to gather data. | A/B testing on a landing page. |
| **Web Scraping** | Extracting data from websites programmatically. | Scraping product prices from e-commerce sites. |
| **APIs** | Accessing data through application interfaces. | Twitter API for tweet data. |
| **Existing Databases** | Using pre-existing records. | Census data, hospital records. |
| **Sensor/IoT Data** | Automated data collection from devices. | Temperature sensors in a factory. |

---

### Q7. What is Data Integration and Data Transformation? Explain with examples.

**Data Integration:**
Combining data from multiple sources into a unified view.

- **Challenge:** Data from different sources may have different formats, schemas, or naming conventions.
- **Example:** Merging customer data from a CRM system (has Customer_ID, Name, Email) with sales data (has Cust_ID, Product, Amount). Requires matching Customer_ID with Cust_ID.

**Techniques:** Entity resolution, schema matching, duplicate detection.

**Data Transformation:**
Converting data from one format or structure to another to make it suitable for analysis.

| Technique | Description | Example |
| :--- | :--- | :--- |
| **Normalization** | Scale values to [0,1] range. | Min-Max scaling: (X−min)/(max−min) |
| **Aggregation** | Summarize data at a higher level. | Daily sales → Monthly totals |
| **Encoding** | Convert categorical to numerical. | Male→1, Female→0 (label encoding) |
| **Discretization** | Convert continuous to bins. | Age: 0-18=Child, 19-60=Adult, 60+=Senior |

---

### Q8. Explain forward fill, backward fill, and interpolation for handling missing values with code.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Day': [1, 2, 3, 4, 5, 6],
    'Temp': [30, np.nan, np.nan, 36, np.nan, 40]
})

# Forward Fill — uses previous value
df['ffill'] = df['Temp'].ffill()
# Result: 30, 30, 30, 36, 36, 40

# Backward Fill — uses next value
df['bfill'] = df['Temp'].bfill()
# Result: 30, 36, 36, 36, 40, 40

# Linear Interpolation — estimates value linearly
df['interpolated'] = df['Temp'].interpolate(method='linear')
# Result: 30, 32, 34, 36, 38, 40

print(df)
```

| Day | Temp | ffill | bfill | interpolated |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 30 | 30 | 30 | 30 |
| 2 | NaN | 30 | 36 | 32 |
| 3 | NaN | 30 | 36 | 34 |
| 4 | 36 | 36 | 36 | 36 |
| 5 | NaN | 36 | 40 | 38 |
| 6 | 40 | 40 | 40 | 40 |

---

### Q9. Explain Symmetric and Asymmetric Binary Attributes with examples.

**Binary Attributes** have only two possible values (0 or 1, True or False).

**Symmetric Binary:**
- Both values (0 and 1) are **equally important**.
- Example: **Gender** (Male = 0, Female = 1) — both values are equally significant.
- When computing similarity, both 0-0 and 1-1 matches are counted.

**Asymmetric Binary:**
- One value is **more important or rare** than the other.
- The rare outcome (usually 1) is of primary interest.
- Example: **HIV Test Result** (Positive = 1, Negative = 0) — a positive result is far more significant than negative.
- When computing similarity, only 1-1 matches matter; 0-0 matches are ignored.

**Jaccard Coefficient** is used for asymmetric binary attributes:
```
J = f₁₁ / (f₀₁ + f₁₀ + f₁₁)
```
Where f₁₁ = both positive, f₀₁ = first 0 second 1, f₁₀ = first 1 second 0.

---

### Q10. Write Python code to read a CSV, filter rows, and save the result to a new CSV.

```python
import pandas as pd

# Read CSV
df = pd.read_csv('employees.csv')
print("Original shape:", df.shape)
print(df.head())

# Filter: employees with salary > 50000 and age > 25
filtered = df[(df['Salary'] > 50000) & (df['Age'] > 25)]
print("\nFiltered records:", len(filtered))
print(filtered.head())

# Sort by salary descending
filtered_sorted = filtered.sort_values('Salary', ascending=False)
print("\nTop 5 highest paid:")
print(filtered_sorted.head())

# Save to new CSV
filtered_sorted.to_csv('high_earners.csv', index=False)
print("\nSaved to high_earners.csv")
```

---

### Q11. Explain the steps involved in building a Machine Learning model.

**Steps for Model Building:**

| Step | Description | Example |
| :--- | :--- | :--- |
| **1. Define Problem** | Clearly state what to predict. | "Predict house prices." |
| **2. Collect Data** | Gather relevant data. | Historical house sales data. |
| **3. Preprocess Data** | Clean, transform, handle missing values. | Remove nulls, normalize prices. |
| **4. Feature Selection** | Choose relevant input features. | Area, bedrooms, location. |
| **5. Split Data** | Divide into training and testing sets. | 80% train, 20% test. |
| **6. Choose Algorithm** | Select appropriate ML model. | Linear Regression for continuous output. |
| **7. Train Model** | Fit the model on training data. | `model.fit(X_train, y_train)` |
| **8. Evaluate Model** | Test on unseen data using metrics. | RMSE, R², MAE on test set. |
| **9. Tune Hyperparameters** | Optimize model settings. | GridSearchCV for best parameters. |
| **10. Deploy Model** | Put into production for real use. | API endpoint for price prediction. |

---

## SECTION C — Long Answer (20 Marks Each, Do Any 2 of 3 = 40 Marks)

---

### Q12. (a) Explain the evolution and applications of Data Science (10 Marks). (b) Describe 5 sources of data with challenges (10 Marks).

**(a) Evolution of Data Science:**

| Era | Development |
| :--- | :--- |
| **1960s** | Statistics and early computing — basic data analysis. |
| **1970s-80s** | Relational databases (SQL) — structured data storage. |
| **1990s** | Data Mining emerges — discovering patterns in databases. |
| **2000s** | Big Data era — Hadoop, distributed computing. |
| **2010s** | Machine Learning & Deep Learning boom — AI-driven insights. |
| **2020s** | AI, AutoML, real-time analytics, LLMs — democratization of Data Science. |

**Applications:**
1. **Healthcare** — Disease prediction, drug discovery, medical imaging.
2. **Finance** — Fraud detection, algorithmic trading, credit scoring.
3. **E-commerce** — Recommendation systems, inventory optimization.
4. **Transportation** — Autonomous vehicles, route optimization (Google Maps).
5. **Agriculture** — Crop yield prediction, precision farming.
6. **Education** — Personalized learning, dropout prediction.

**(b) Five Sources of Data:**

| Source | Example | Challenges |
| :--- | :--- | :--- |
| **Time Series** | Stock prices, weather data | Handling seasonality, trends, missing timestamps |
| **Transactional** | Bank records, e-commerce orders | High volume, real-time processing, consistency |
| **Biological** | Genomic data, patient records | Privacy (HIPAA), complex structures, large size |
| **Spatial** | GPS data, satellite imagery | Coordinate systems, large files, real-time needs |
| **Social Network** | Tweets, posts, comments | Unstructured text, noise, bots, ethics, API limits |

---

### Q13. Write a complete Python program to analyse a dataset using Pandas, NumPy, and Matplotlib.

**Scenario:** File `library_books.csv` has: `Book_ID, Title, Author, Genre, Pages, Rating, Price, Copies_Sold`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the dataset
df = pd.read_csv('library_books.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# 2. Basic statistics
print("\nMean Rating:", df['Rating'].mean())
print("Median Pages:", df['Pages'].median())
print("Most common Genre:", df['Genre'].mode()[0])

# 3. Missing values
print("\nMissing values:")
print(df.isnull().sum())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Price'] = df['Price'].fillna(df['Price'].median())

# 4. Filter top-rated books
top_books = df[df['Rating'] >= 4.5]
print(f"\nBooks with Rating >= 4.5: {len(top_books)}")
print(top_books[['Title', 'Rating', 'Copies_Sold']])

# 5. Group by Genre
genre_stats = df.groupby('Genre').agg({
    'Rating': 'mean',
    'Copies_Sold': 'sum',
    'Price': 'mean'
}).round(2)
print("\nGenre Statistics:")
print(genre_stats)

# 6. Bar chart — Average rating by genre
plt.figure(figsize=(10, 5))
genre_stats['Rating'].plot(kind='bar', color='teal', edgecolor='black')
plt.title('Average Rating by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Scatter plot — Pages vs Rating
plt.figure(figsize=(8, 5))
plt.scatter(df['Pages'], df['Rating'], alpha=0.6, color='purple')
plt.title('Pages vs Rating')
plt.xlabel('Number of Pages')
plt.ylabel('Rating')
plt.grid(True, alpha=0.3)
plt.show()

# 8. Save
df.to_csv('library_cleaned.csv', index=False)
print("Saved cleaned data!")
```

---

### Q14. Explain Supervised, Unsupervised, Semi-Supervised, and Reinforcement Learning with examples.

**(a) Supervised Learning (5 Marks):**
- **Labelled data** (input-output pairs provided).
- Model learns the mapping from input → output.
- **Types:** Classification (predicting categories), Regression (predicting numbers).
- **Algorithms:** k-NN, Decision Tree, Linear Regression, SVM.
- **Example:** Email spam detection — model trained on emails labelled as spam/not spam.

**(b) Unsupervised Learning (5 Marks):**
- **Unlabelled data** (only inputs, no outputs).
- Model discovers hidden patterns or groupings.
- **Types:** Clustering, Association Rule Mining, Dimensionality Reduction.
- **Algorithms:** K-Means, DBSCAN, Apriori, PCA.
- **Example:** Customer segmentation — grouping users by purchasing behaviour without predefined categories.

**(c) Semi-Supervised Learning (5 Marks):**
- Combination of **small amount of labelled data + large amount of unlabelled data**.
- Model uses labelled data for supervised learning and leverages unlabelled data for better generalization.
- Useful when labelling data is expensive or time-consuming.
- **Algorithms:** Self-training, Label Propagation.
- **Example:** Medical image classification — only a few X-rays are labelled by doctors, but thousands of unlabelled X-rays are available.

**(d) Reinforcement Learning (5 Marks):**
- **Agent** learns by interacting with an **environment** and receiving **rewards/penalties**.
- No labelled data; model learns through trial and error.
- Goal: maximize cumulative reward.
- **Key Concepts:** Agent, State, Action, Reward, Policy.
- **Algorithms:** Q-Learning, Deep Q-Network (DQN), Policy Gradient.
- **Example:** AlphaGo — Google's AI learned to play Go by playing millions of games against itself, receiving rewards for winning.

**Comparison Table:**

| Aspect | Supervised | Unsupervised | Semi-supervised | Reinforcement |
| :--- | :--- | :--- | :--- | :--- |
| **Data** | Labelled | Unlabelled | Mix | No data (environment) |
| **Feedback** | Direct (correct answer) | None | Partial | Reward/penalty |
| **Goal** | Predict | Discover patterns | Predict with less labels | Maximize reward |
| **Example** | Spam filter | Customer groups | Medical imaging | Game AI |

---

**— End of Paper 3 (Easy) —**
