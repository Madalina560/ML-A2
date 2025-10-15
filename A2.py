import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Patch

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

# # cross validation of lasso regression
cVals = [0.001, 0.1, 1, 10, 100] # values for alpha needed to train Lasso regression
xConcat = pd.concat([X1, X2], axis = "columns") # concatenate X1 and X2 to pass them into lasso regression function
xConcat.columns = ["X1", "X2"]
# polyLassoReg = PolynomialFeatures(degree=5) # setting up poly lasso regression
# XTrainPoly = polyLassoReg.fit_transform(xConcat) # convert X1, X2 to work w/ polynomial lasso regression

# modelStore = {}

# for c in cVals:
#     polyLassoModel = Lasso(alpha = 1/(2*c))
#     polyLassoModel.fit(XTrainPoly, Y)
#     modelStore[c] = polyLassoModel # storing the model for later use
#     feature_names = polyLassoReg.get_feature_names_out(["X1", "X2"])
#     coeffs = polyLassoModel.coef_
#     print("C: ", c)
#     print("Feature names: ", feature_names)
#     print("Coefficients: ", coeffs)

# # i) c) Generate Predictions & Plot
# xTest = []
# grid = np.linspace(-2, 2, num = 20) # smaller amount of points so i can run it easier
# for i in grid:
#     for j in grid:
#         xTest.append([i, j])
# xTest = np.array(xTest)

# xTestPoly = polyLassoReg.transform(xTest)

# colors = ["red", "yellow", "green", "blue", "purple"]
# surfPatches = []

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for idx, (c, model) in enumerate(modelStore.items()):
#     yPred = model.predict(xTestPoly)
#     x1Grid, x2Grid = np.meshgrid(grid, grid)
#     yPredSurf = yPred.reshape(x1Grid.shape)
#     surfPatches.append(Patch(color=colors[idx], label = f"Prediction C = {c}"))
#     ax.plot_surface(x1Grid, x2Grid, yPredSurf, alpha = 0.2, color =colors[idx])

# scatPatch = Patch(color="red", label = "Training data")

# ax.scatter(X[:,0], X[:,1], Y, color = "red", label = "Training data")
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_zlabel("Y Predictions")
# ax.legend(handles=[scatPatch] + surfPatches)
# plt.title("Predicted data vs Training Data")
# plt.show()

# i) e) Repeat b - c for Ridge Regression
polyRidgeReg = PolynomialFeatures(degree = 5)
XTrainPolyR = polyRidgeReg.fit_transform(xConcat)
modelStoreR = {}

for c in cVals:
    polyRidgeModel = Ridge(alpha = 1/(2*c))
    polyRidgeModel.fit(XTrainPolyR, Y)
    modelStoreR[c] = polyRidgeModel
    featureNameR = polyRidgeReg.get_feature_names_out(["X1", "X2"])
    coeffR = polyRidgeModel.coeff_
    print("C: ", c)
    print("Feature names: ", featureNameR, "\n")
    print("Coeeficients: ", coeffR, "\n") # get vals & put into table tomorrow


# for later when you need to do cross validation
# meanErr = []
# meanErr1 = []
# stdErr = []
# stdErr1 = []

# for c in cVals:
#     polyLassoModel = Lasso(alpha=1/(2*c))
#     temp = []; temp1 = []
#     kf = KFold(n_splits=5)
#     for train, test in kf.split(XTrainPoly):
#         polyLassoModel.fit(XTrainPoly[train], Y[train])

#         ypredTest = polyLassoModel.predict(XTrainPoly[test])
#         temp.append(mean_squared_error(Y[test], ypredTest))

#         ypredTrain = polyLassoModel.predict(XTrainPoly[train])
#         temp1.append(mean_squared_error(Y[train], ypredTrain))
#     meanErr.append(np.array(temp).mean()); stdErr.append(np.array(temp).std())
#     meanErr1.append(np.array(temp1).mean()); stdErr1.append(np.array(temp1).std())
    # scores = cross_val_score(polyLassoModel, XTrainPoly, Y, cv = 5, scoring="neg_mean_squared_error")
    # meanScores.append(scores.mean())
    # stdScores.append(scores.std())
    # print("Alpha: ", c, "\n")
    # print("Cross Validation Scores: ",scores, "\n")
    # print("Accuracy: ", scores.mean(), "STD Dev: ", scores.std(), "\n")

# plt.figure()
# plt.errorbar(cVals, meanErr, yerr = stdErr)
# plt.errorbar(cVals, meanErr1, yerr = stdErr1)
# plt.xlabel("C level")
# plt.ylabel("Mean Square Error")
# plt.legend(["Test data", "Training data"])
# plt.title("Cross-Validation Error vs C")
# plt.show()