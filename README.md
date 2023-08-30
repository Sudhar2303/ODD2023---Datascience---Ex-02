# Ex02-Outlier
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

(i) Using IQR detect weight outliers and print them

(ii) Using IQR, detect height outliers and print them
# Aim:
TO detect and remove the outliers in the given data set and save the final data.

# EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

# ALGORITHM
STEP 1
Read the given Data

STEP 2
Get the information about the data

STEP 3
Detect the Outliers using IQR method and Z score

STEP 4
Remove the outliers

# CODE AND OUTPUT
```
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/6876b7c5-f0ee-40f7-9fc2-84fbad9ac543)
```
from google.colab import files
uploaded = files.upload()
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/fda61b9c-4219-451c-be50-13317010f500)
```
df = pd.read_csv("bhp.csv")
q1 = df['price_per_sqft'].quantile(0.25)
q2 = df['price_per_sqft'].quantile(0.5)
q3 = df['price_per_sqft'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/3011709e-5c9e-4441-8740-93b3120e10fc)
```
low = q1-1.5*iqr
low
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/a4dbf790-962f-4275-aea6-4a439d629be6)
```
high = q3+1.5*iqr
high
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/b05f8f75-5f62-458f-88fc-844a878094a2)
```
df = df[((df['price_per_sqft']>=low) & (df['price_per_sqft']<=high))]
df
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/c797b523-bd2f-4a3f-8126-1722efb5d52d)
```
z = np.abs(stats.zscore(df['price_per_sqft']))
z
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/f1155e1f-8d81-4b50-949b-9a34dec2ac90)
```
df1 = df[z<3]
df1
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/09b8b545-36fe-4ba5-8926-92c7e2f191ca)

```
from google.colab import files
uploaded = files.upload()
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/02b18877-745d-4906-8ac5-4422dd70079f)

```
df = pd.read_csv("height_weight.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/bf4ba315-d278-41f1-86f9-bc6729f19638)

```
low = q1 - 1.5*iqr
low
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/1965472e-b309-4fda-9127-2d0934ef4f4c)

```
high = q3+1.5*iqr
high
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/128c5362-506f-4f60-9287-a6a52a56c49a)
```
df = df[((df['height'] >=low) & (df['height']<= high))]
df
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/3482b871-e4c9-4fb5-87bb-54264d6d6d6d)

```
z = np.abs(stats.zscore(df['height']))
z
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/f3129fa4-a20b-40af-8e39-5062dfd27cd0)

```
df1 = df[z<3]
df1
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/d78445fe-6351-4ff7-a59f-9e5f0a674fd3)

```
df = pd.read_csv("height_weight.csv")
q1 = df['weight'].quantile(0.25)
q2 = df['weight'].quantile(0.5)
q3 = df['weight'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/50779604-9983-43bd-bc8c-2120a765d903)

```
low = q1 - 1.5*iqr
low
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/7cb9a9b5-edfc-44c1-9590-658cfc579f52)

```
high = q3 + 1.5*iqr
high
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/e8105ce0-5d96-4932-8fcd-bc72a640312b)

```
df1 = df[((df['weight'] >=low) & (df['weight']<= high))]
df1
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/c5eca23d-ffb9-4db1-a6a5-c7d237c5f835)

```
z = np.abs(stats.zscore(df1['weight']))
z
```

![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/35e3516b-86dc-4c14-bc1b-270782015bba)

```
df2 = df1[z<3]
df2
```

![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/94f013bd-7666-4fd9-83a9-6cbfb3ca29f2)

```
from google.colab import files
uploaded = files.upload()
```

![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/86758662-c1d5-42b7-bf38-eee0b7ad7059)

```
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/dcc914e9-cff0-428e-b7c9-3ef1ff845b04)

```
low = q1 - 1.5*iqr
low
```

![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/5d7d7102-3260-4184-ab20-0dd3e3217058)

```
high = q3 + 1.5*iqr
high
```

![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/56fb6ffd-9b5f-4602-bf4f-c6d315fb95a7)

```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```

![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/9f435b01-6ecd-4bdb-af7c-bd78e2616d28)

```
z = np.abs(stats.zscore(df['height']))
z
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/a4b575d8-cbc9-499f-ac32-c378bce2caad)

```
df1 = df[z<3]
df1
```
![image](https://github.com/Sudhar2303/ODD2023---Datascience---Ex-02/assets/133684710/90278c64-48f4-4f2c-843a-d3528bc67c7a)

# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.
