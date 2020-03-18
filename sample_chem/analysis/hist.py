import pandas as pd
import matplotlib.pyplot as plt
import sys
from pylab import rcParams

df1 = pd.read_csv(sys.argv[1])#ave
df2 = pd.read_csv(sys.argv[2])#PLIF
df3 = pd.read_csv(sys.argv[3])#nonPLIF
df4 = pd.read_csv(sys.argv[4])#total

total_ave = df1.iloc[:, 0]
PLIF_ave = df1.iloc[:, 1]
nonPLIF_ave = df1.iloc[:, 2]
PLIF = df2.iloc[:, 0]
nonPLIF = df3.iloc[:, 0]
Total = df4.iloc[:, 0]

ave_labels = ["IG_total_average", "IG_nonPLIF_average", "IG_PLIF_average"]
labels = ["IG_Total", "IG_nonPLIF", "IG_PLIF"]

rcParams["figure.figsize"] = 10, 10
plt.hist([total_ave, nonPLIF_ave, PLIF_ave], label=ave_labels, density=True, bins=50, stacked=False, color=["green", "blue", "red"], alpha=0.7)
plt.legend()
plt.xlabel("IG_average")
plt.ylabel("density")
plt.title("IG average histogram")

plt.show()

rcParams["figure.figsize"] = 10, 10
plt.hist(total_ave, label=ave_labels[0], density=True, bins=50, stacked=False, color="green", alpha=0.7)
plt.hist(nonPLIF_ave, label=ave_labels[1], density=True, bins=50, stacked=False, color="blue", alpha=0.5)
plt.hist(PLIF_ave, label=ave_labels[2], density=True, bins=50, stacked=False, color="red", alpha=0.3)

plt.legend()
plt.xlabel("IG_average")
plt.ylabel("density")
plt.title("IG average histogram")

plt.show()


rcParams["figure.figsize"] = 10, 10
plt.hist([Total, nonPLIF, PLIF], label=labels, density=True, bins=100, stacked=False, color=["green", "blue", "red"], alpha=0.7)
plt.legend()
plt.xlabel("IG")
plt.ylabel("density")
plt.title("IG histogram")

plt.show()

rcParams["figure.figsize"] = 10, 10
plt.hist(Total, label=labels[0], density=True, bins=100, stacked=False, color="green", alpha=0.7)
plt.hist(nonPLIF, label=labels[1], density=True, bins=100, stacked=False, color="blue", alpha=0.5)
plt.hist(PLIF, label=labels[2], density=True, bins=100, stacked=False, color="red", alpha=0.3)

plt.legend()
plt.xlabel("IG")
plt.ylabel("density")
plt.title("IG histogram")

plt.show()
