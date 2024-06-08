"""About Dataset
This dataset contains information about the salaries of employees at a company. Each row represents a different
employee, and the columns include information such as age, gender, education level, job title, years of experience,
and salary.
Columns:
Age: This column represents the age of each employee in years. The values in this column are numeric.
Gender: This column contains the gender of each employee, which can be either male or female. The values in this
column are categorical.
Education Level: This column contains the educational level of each employee, which can be high school, bachelor's
degree, master's degree, or PhD. The values in this column are categorical.
Job Title: This column contains the job title of each employee. The job titles can vary depending on the company and
may include positions such as manager, analyst, engineer, or administrator. The values in this column are categorical.
Years of Experience: This column represents the number of years of work experience of each employee. The values in
this column are numeric.
Salary: This column represents the annual salary of each employee in US dollars. The values in this column are
numeric and can vary depending on factors such as job title, years of experience, and education level.
** The purpose of creating this dataset is solely for educational use, and any commercial use is strictly prohibited
and this dataset was large language models generated and not collected from actual data sources."""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# df = sns.load_dataset("Salary Data")
df = pd.read_csv("Salary Data.csv")

# data profiling and analysis
print(df.info())
'''upon observing, we can say that out of 375 entries, 373 are notnull, meaning 2 rows has null data'''

# numerical data analysis
print(df.describe())

# categorical data analysis
print(df.describe(include = 'object'))

# finding null(NaN) data
print(df.isna().sum())
''' this confirms that in each column, there are 2 missing data ie NaN'''

# showing NaN data in heatmap
sns.heatmap(df.isna())
print(plt.show())
'''upon visualizing the data in heatmap, we can see white lines around 165 and 255, meaning null data is around there'''

# locating the NaN data
print(df.loc[df.isna().any(axis = 'columns')])
'''with this we can see that row 172 and 260 has missing data'''

'''Missing data Handling
### Deletion ###
row: if target or label(y) is missing; if entire row is missing we drop the row
column: If a column contains >70% data missing we drop column.
### Imputation (fill in missing value) ###
mean: data (column) has no outliers and normally distributed --use mean to fill in missing data
median: if data(column) has outlier use median to fill in missing data
mode: if data(column) is categorical and data is missing we use mode to fill in missing data.
### Mathematical Technique ###
Interpolation and Extrapolation
### Algorithms ###
MICE
Iterative Impute
LinearRegression
RandomForest
KNN
### EDA and Domain knowledge ###
////In our case entire row is missing so we drop the rows.////'''

# deleting missing data as y label or output or dependant variable ie Salary is missing in 2 rows
df.dropna(inplace = True)  # dropping missing dependant variable and using inplace to make permanent change to DataFrame
print(df.isna().sum())  # checking the changes made

# Exploratory Data Analysis (EDA)
# histogram
sns.histplot(df["Salary"], kde = True, bins = 6)
print(plt.show())

# boxplot to check if we have any outliers.
sns.boxplot(df['Salary'])
print(plt.show())
'''No outliers says we can use mean to fill NaN data, otherwise use median'''

'''Now lets say we want to predict the salary. Salary is a dependent variable and continuous data.
Hence, we use linear regression for this data type.'''

# checking correlation of each datatype

# using heatmap to check correlation of all numeric variables
sns.heatmap(df.corr(numeric_only = True), annot = True)
print(plt.show())
'''The above heatmap shows Salary is positively correlated with Age and Years of Experience with values 0.92 and 0.93 
respectively.
The plot also shows that there is correlation between Age and Years of Experience.
In our data we want to predict Salary using Age or Years of Experience. Hence, Salary is a dependent variable and 
Age, Years of Experience is independent variables.
The assumption of Linear Regression "Independence " stats that the independent variable should not be correlated. In 
our case, Years of Experience and Age are independent variable but they are correlated. This is a problem of 
multi-collinearity. In such a case, we have to drop one column.
Which one to drop? We should drop Age column as it is a readily available in inference time. It is easier to ask 
Years of Experience instead of Age.'''

# using scatterplot to visualize correlation between YoE and Salary
sns.scatterplot(x = df['Years of Experience'], y = df['Salary'])
print(plt.show())

# selecting dependent(y) and independent(X) variable
X = df.loc[:, ['Years of Experience']]  # rows all and column YoE. It should be in 2D/DataFrame
y = df['Salary']  # y should always be in 1D/Series
sns.scatterplot(x = X['Years of Experience'], y = y)
print(plt.show())

sns.pairplot(df, hue = "Gender")
print(plt.show())
