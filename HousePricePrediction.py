
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


# %% CHECK FOR DUPLICATE ROWS
train.duplicated().sum()

# %% FIND CORRELATION
train.corr()

# %%
train[numerical_cols].isna().sum()

# %% 
X = train.drop(['SalePrice'], axis=1)      # Features
y = train[['SalePrice']].values.ravel()    # Target variable


# %% DATA PROCESSING
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

for c in categorical_cols:      
    lbl = LabelEncoder() 
    lbl.fit(list(X[c].values)) 
    X[c] = lbl.transform(list(X[c].values))

integer_transformer = Pipeline(steps = [
   ('imputer', SimpleImputer(strategy = 'mean')),   
   ('scaler', StandardScaler())])                  



categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))  
])

preprocessor = ColumnTransformer(                    
   transformers=[
       ('ints', integer_transformer, numerical_cols),
       ('cat', categorical_transformer, categorical_cols)])


X = preprocessor.fit_transform(X)                        


# %% DATA SPLIT
from sklearn.model_selection import train_test_split  

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.20, random_state=42,shuffle=False)
print('X_train:',X_train.shape,'y_train:',y_train.shape,'\nX_valid :',X_valid.shape,'y_valid:',y_valid.shape)


# %% GradientBoostingRegressor Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold,cross_val_score

kf = KFold(n_splits =10, shuffle = True, random_state = 100) 

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(X_train,y_train)

scores = cross_val_score(GBoost,X_train, y_train, scoring="neg_mean_squared_error", cv = kf)   
scores = np.sqrt(-scores)

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())

# %% TEST DATA
test_ID = test['Id'] 

test.drop(columns=['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'PoolArea', 'Fence', 'MiscFeature'], inplace=True)


for c in categorical_cols:
    lbl = LabelEncoder() 
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))

X_test = preprocessor.fit_transform(test)   

prediction = GBoost.predict(X_test)   #making the prediction of the test data

print(prediction)


# %% SUBMISSION
submission = DataFrame()    #create an empty dataframe
submission['Id'] = test_ID     #create a column with the test Id's
submission['SalePrice'] = prediction   #create a column with the test predictions
submission.to_csv('submission.csv',index=False) #submission file
submission.head()

# %%
