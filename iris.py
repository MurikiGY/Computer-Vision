import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# import some data to play with
iris = datasets.load_iris()

print ("DATA")
print (iris.data)
print ("TARGET")
print (iris.target)

iris_df = pd.DataFrame(data = iris.data)
iris_df['target'] = iris.target

iris_df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']
print ("DESCRIBING PANDAS DF")
print (iris_df.describe())
print ("HEAD")
print (iris_df.head())


# Plot the training points
plt.scatter(iris_df[iris_df.columns[0]], iris_df[iris_df.columns[1]],c= iris_df['target'], edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()

plt.scatter(iris_df[iris_df.columns[2]], iris_df[iris_df.columns[3]],c= iris_df['target'], edgecolor='k')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.show()

