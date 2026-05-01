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
axes[1, 1].pie(counts, labels=unique, autopct='%1.1f%%', colors=['#e74c3c', '#f39c12', '#2ecc71',
'#3498db'])
axes[1, 1].set_title('Rating Distribution')
plt.tight_layout()
plt.savefig('dashboard.png', dpi=150)
plt.show()