import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

# i) a) Plot data on scatter plot (given in assignment sheet)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y, label = "points")
plt.title("Visualisation of raw data points in a 3D space")
plt.xlabel("Column 0 data (X1)")
plt.ylabel("Column 1 data (X2)")
plt.legend(loc = "upper right", bbox_to_anchor =(1.2, 1.2))
plt.show()