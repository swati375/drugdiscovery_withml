"""# Part 5: **compare regression models**

comparing several ML algorithms for build regression models of sars cov2 inhibitors.
"""

! pip install lazypredict

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyRegressor

from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)

df=pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/data_ml/bioactivity_data_3class_pIC50_pubchem_fp.csv')

X = df.drop('pIC50', axis=1)
Y = df.pIC50

print(X.shape,Y.shape)

from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)
X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)#, random_state=42)

"""**Compare ML algorithms**"""

clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test=clf.fit(X_train,X_test,Y_train,Y_test)
# models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
# models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

train

test

"""**Data visualization of model performance**"""

# Bar plot of R-squared values
import matplotlib.pyplot as plt
import seaborn as sns

#train["R-Squared"] = [0 if i < 0 else i for i in train.iloc[:,0] ]

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train.index, x="R-Squared", data=train)
ax.set(xlim=(0, 1))

# Bar plot of RMSE values
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train.index, x="RMSE", data=train)
ax.set(xlim=(0, 10))

# Bar plot of calculation time
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train.index, x="Time Taken", data=train)
ax.set(xlim=(0, 10))

