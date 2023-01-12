from seaborn import load_dataset, pairplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from seaborn import scatterplot

df = load_dataset('penguins')
print(df.head())
df = df.dropna()
print(len(df))
pairplot(df, hue='species')
plt.show()
X = df[['bill_length_mm', 'bill_depth_mm']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

#testing accuracy
clf = SVC(kernel='linear')
clf.fit(X_train, y_train) 
predictions = clf.predict(X_test)
print(predictions[:5])



w = clf.coef_[0]
b = clf.intercept_[0]
x_visual = np.linspace(32,57)
y_visual = -(w[0] / w[1]) * x_visual - b / w[1]

scatterplot(data = X_train, x='bill_length_mm', y='bill_depth_mm', hue=y_train)
plt.plot(x_visual, y_visual)
plt.show()
print(accuracy_score(y_test, predictions))

