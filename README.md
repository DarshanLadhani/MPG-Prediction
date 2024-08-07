In this project, we successfully developed a regression model to predict the miles per gallon (mpg) of vehicles using various features such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin.

## **We have performed following tasks.**

### 1.Library Imports and Data Import
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/MPG.csv')

```
## 2.Data Exploration


```
data.head()
```

## 3.Data Preprocessing

*   Handling Missing Values
```
data = data.dropna()
```
*   Data Standarization
```
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
```

## 4.Data Visualization


```
sns.pairplot(data = data , x_vars=['displacement','horsepower','weight','acceleration', 'mpg'],y_vars=['mpg'])
sns.regplot(x = 'displacement', y = 'mpg', data = data)
```

## 5.Model Development

```
model = LinearRegression()
```

## 6.Model Training and Evaluation

*   Train-Test Split: Splitting the data into training and testing sets.
*   Model Training: Training the regression model on the training data.
*  Model Evaluation: Evaluating the model performance using metrics such as Mean Squared Error (MSE), R-squared, etc.



```
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 252)
model = LinearRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
mean_absolute_percentage_error(y_test , y_pred)
mean_absolute_error(y_test , y_pred)
r2_score(y_test , y_pred)

```

