import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# id:15--15-15 

# import data (week3.csv)
df = pd.read_csv("week3.csv", skiprows=1) # skip ID row
print("Shape of df: ", df.shape)
print("Columns: ", df.columns)
print(df.head())
X1 = df.iloc[:,0] # exctracting all rows from column 0
X2 = df.iloc[:,1] # extracting all rows from column 1
X = np.column_stack((X1, X2))
Y = df.iloc[:,2] # extracting all rows from column 2, want to plot this differenciating between +/-1

# i) 
# a) Plot data on scatter plot (given in assignment sheet)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y, label = "points")
plt.title("Visualisation of raw data points in a 3D space")
plt.xlabel("Column 0 data (X1)")
plt.ylabel("Column 1 data (X2)")
plt.legend(loc = "upper right", bbox_to_anchor =(1.2, 1.2))
plt.show()

# b) Train Lasso Regression w/ polynomial features
# alpha in lasso is inversely proportional to C, i.e when alpha = 0.1, C = 10

# cross validation of lasso regression
alphaVals = [0.001, 0.1, 1, 10, 100] # values for alpha needed to train Lasso regression
xConcat = pd.concat([X1, X2], axis = "columns") # concatenate X1 and X2 to pass them into lasso regression function
polyLassoReg = PolynomialFeatures(degree=5) # setting up poly lasso regression
XTrainPoly = polyLassoReg.fit_transform(xConcat) # convert X1, X2 to work w/ polynomial lasso regression

for alpha in alphaVals:
    polyLassoModel = Lasso(alpha=alpha)
    scores = cross_val_score(polyLassoModel, XTrainPoly, Y, cv = 5, scoring="neg_mean_squared_error") 
    print("Alpha: ", alpha, "\n")
    print("Cross Validation Scores: ",scores, "\n")
    print("Accuracy: ", scores.mean(), "STD Dev: ", scores.std(), "\n")