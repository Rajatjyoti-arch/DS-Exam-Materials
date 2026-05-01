# Introduction to Data Science — Final Examination

**Course Code:** BCSE2C05 / BCCS2C01 / BMNC2C01
**Semester:** II
**Maximum Marks:** 100 | **Time:** 3 Hours

> **Instructions:**
> - All questions carry marks as indicated.
> - Assume suitable data wherever necessary.
> - Figures, code snippets, and diagrams should be clear and well-labelled.

---

## SECTION A — Short Answer Questions (5 Marks Each)

**Attempt any 4 out of 5. (4 × 5 = 20 Marks)**

| Q. No. | Question | Unit | CO |
| :---: | :--- | :---: | :---: |
| 1. | Define Nominal, Ordinal, and Binary attributes with one example of each. How do they differ from Numeric attributes? | II | 2 |
| 2. | What is Data Normalization? Explain Min-Max normalization with a suitable numerical example. | III | 3 |
| 3. | Differentiate between Supervised and Unsupervised Learning. List two algorithms that belong to each category. | II | 4 |
| 4. | What are the different techniques used in Data Cleaning? Explain how handling missing values and noisy data improves the quality of a dataset. | II | 1, 2 |
| 5. | What is a CSV file? Write a Python code snippet to read a CSV file using Pandas, display its shape, and list all column names. | III | 3 |

---

## SECTION B — Analytical / Application Questions (8 Marks Each)

**Attempt any 5 out of 6. (5 × 8 = 40 Marks)**

| Q. No. | Question | Unit | CO |
| :---: | :--- | :---: | :---: |
| 6. | Explain the **k-Nearest Neighbours (k-NN)** algorithm with a suitable example. Discuss how the choice of *k* affects classification accuracy and mention at least two distance metrics used in k-NN. | II | 4 |
| 7. | A retail dataset contains columns: `Product_ID`, `Category`, `Price`, `Quantity_Sold`, `Rating`, `Return_Status`. Using **Pandas**, write Python code to:<br>(i) Filter all products with `Rating > 4.0` and `Quantity_Sold > 100`.<br>(ii) Group the data by `Category` and compute the average `Price` for each category.<br>(iii) Sort the result in descending order of average price. | III | 3 |
| 8. | What is **Data Reduction**? Explain any three techniques of data reduction (e.g., Dimensionality Reduction, Numerosity Reduction, Data Compression) with examples. Why is data reduction important before applying machine learning models? | II | 2 |
| 9. | Compare **Histograms, Boxplots, and Scatterplots**. For each, explain when it is best used and write a Python code snippet using Matplotlib/Seaborn to generate any one of them on sample data. | III | 3 |
| 10. | Describe the **K-Means Clustering** algorithm step-by-step. Given the 1-D data points `{2, 4, 10, 12, 3, 20, 30, 11, 25}` and initial centroids `C1 = 2` and `C2 = 4`, perform the first two iterations of K-Means and show the cluster assignments after each iteration. | II | 4 |
| 11. | What are **Ethical Considerations** in Data Science? Discuss issues related to data privacy, algorithmic bias, and informed consent. Illustrate with a real-world example where unethical data practices led to negative consequences. | II | 1, 2 |

---

## SECTION C — Long Answer / Case-Based Questions (20 Marks Each)

**Attempt any 2 out of 3. (2 × 20 = 40 Marks)**

| Q. No. | Question | Unit | CO |
| :---: | :--- | :---: | :---: |
| 12. | **(a)** Explain the complete **Data Science Lifecycle** with a detailed diagram. Discuss each phase — Problem Definition, Data Collection, Data Cleaning, Exploratory Data Analysis, Modelling, Evaluation, and Deployment — with suitable examples. **(10 Marks)**<br><br>**(b)** Describe the concept of **Big Data** and explain its defining characteristics (5 V's). Discuss at least five **sources of data** (Time Series, Transactional, Biological, Spatial, Social Network) with real-world use cases and the challenges involved in collecting each type. **(10 Marks)** | I | 1, 2 |
| 13. | A CSV file named **`employee_performance.csv`** contains data of 500 employees of an IT company. The dataset contains the following fields:<br>`Emp_ID, Name, Age, Gender, Department, Experience_Years, Monthly_Salary, Performance_Score, Projects_Completed, Joining_Date, Last_Promotion_Date`<br><br>Write a complete Python program using **NumPy, Pandas, Matplotlib, and Seaborn** to perform:<br>1. Read the CSV file and display its shape, data types, and first 5 rows.<br>2. Compute the mean, median, and mode of `Performance_Score`.<br>3. Find and print the number of missing values in each column. Replace missing values in `Monthly_Salary` with the median and `Performance_Score` with the mean.<br>4. Create a new column `Salary_Category` — "Low" if salary < 30000, "Medium" if 30000–60000, "High" if > 60000.<br>5. Plot a **boxplot** for `Monthly_Salary` grouped by `Department` and a **histogram** for `Performance_Score`.<br>6. Calculate the **IQR** of `Monthly_Salary` and identify any outliers.<br>7. Save the cleaned and processed dataset as **`employee_cleaned.csv`**. | II, III | 2, 3 |
| 14. | **(a)** Compare and contrast **Descriptive and Predictive Data Mining** with suitable examples. Explain how each type is applied in the context of a real-world business scenario (e.g., e-commerce recommendation system). **(8 Marks)**<br><br>**(b)** A university wants to predict student dropout rates using machine learning. Describe the **end-to-end machine learning pipeline** you would build — from data collection and preprocessing to model selection, training, evaluation, and deployment. Discuss which ML approach (Supervised, Semi-supervised, or Reinforcement Learning) is most appropriate and justify your choice. Include a discussion on feature engineering, handling class imbalance, and model evaluation metrics (accuracy, precision, recall, F1-score). **(12 Marks)** | I, II | 1, 4 |

---

**— End of Question Paper —**
