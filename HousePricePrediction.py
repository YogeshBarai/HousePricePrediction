
# %% IMPORTS
from pandas import read_csv, concat, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %% READ DATA
train = read_csv('./data/train.csv')
test = read_csv('./data/test.csv')


# %% EDA
train.describe()
train.dtypes.value_counts()


# %% CHECK NULL VALUES
total = train.isnull().sum().sort_values(ascending=False)
count = train.isnull().count().sort_values(ascending=False)

percentage = total / count

missing_data = concat([total, percentage], axis=1, keys=['Total', 'Percentage'])
print(missing_data)

# %% PLOT COLUMNS WITH NULL VALUES
train.isna().sum()[train.isna().sum()>0].plot(kind='bar')

# %% DROP COLUMNS THAT HAVE MORE NULL VALUES
# DROPPING ID AND PoolArea (Linked to PoolQC)
train.drop(columns=['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'PoolArea', 'Fence', 'MiscFeature'], inplace=True)


# %% VARIOUS COLUMN INFORMATION
df_info = DataFrame(
    data={
        'Missing Values Count': train.isna().sum(),
        'Unique Values Count': train.nunique(),
        'Unique Values': [train[col].unique().tolist() for col in train.columns],
        'Column type': train.dtypes
        })

df_info.head(74)


# %% TARGET COLUMN
target = 'SalePrice'

# %% NUMERICAL & CATEGORICAL COLUMNS
numerical_cols = (train.select_dtypes(include=['int64', 'float64']).columns.tolist())
categorical_cols = (train.select_dtypes(exclude=['int64', 'float64']).columns.tolist())


# %% PLOT CHARTS
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
plt.figure(figsize=(24,60))
col_count = 1

for col in numerical_cols[:37]:
    plt.subplot(8,5,col_count)
    sns.distplot(x=train[col], kde=False, bins=10)
    plt.title(f'Histogram for {col}')
    
    col_count += 1
# %%
