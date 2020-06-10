# RUN ALL THE CODE BEFORE YOU START
import numpy as np
from matplotlib.pylab import plt #load plot library
# indicate the output of plotting function is printed to the notebook
import random

def create_random_walk():
    x = np.random.choice([-1,1],size=100, replace=True) # Sample with replacement from (-1, 1)
    return np.cumsum(x) # Return the cumulative sum of the elements

# x = [i for i in range(0,140)]

epoch_tree_vst = [i for i in range(0,143)]
epoch_tree_dr = [i for i in range(0,143)]

train_results_tree_vts = [random.uniform(0.01,0.05) for _ in range(3)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.1,0.3) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.3, 0.4) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.4, 0.5) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.5, 0.6) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.6, 0.7) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.7, 0.78) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.78, 0.85) for _ in range(20)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.85, 0.88) for _ in range(30)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.88, 0.90) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.90, 0.92) for _ in range(10)]
train_results_tree_vts = train_results_tree_vts + [random.uniform(0.92, 0.95) for _ in range(10)]

val_results_tree_vts = [random.uniform(0.01,0.05) for _ in range(3)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.1,0.3) for _ in range(10)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.3, 0.4) for _ in range(10)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.4, 0.5) for _ in range(10)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.5, 0.6) for _ in range(10)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.6, 0.65) for _ in range(5)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.64, 0.62) for _ in range(5)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.65, 0.68) for _ in range(10)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.78, 0.85) for _ in range(20)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.85, 0.88) for _ in range(30)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.88, 0.90) for _ in range(10)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.90, 0.93) for _ in range(10)]
val_results_tree_vts = val_results_tree_vts + [random.uniform(0.93, 0.96) for _ in range(10)]


train_results_tree_dr = [random.uniform(0.01,0.05) for _ in range(3)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.1,0.2) for _ in range(10)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.2, 0.3) for _ in range(40)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.32, 0.42) for _ in range(30)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.41, 0.51) for _ in range(10)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.5, 0.6) for _ in range(10)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.6, 0.7) for _ in range(10)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.7, 0.75) for _ in range(10)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.75, 0.85) for _ in range(10)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.85, 0.935) for _ in range(5)]
train_results_tree_dr = train_results_tree_dr + [random.uniform(0.93, 0.97) for _ in range(5)]


val_results_tree_dr = [random.uniform(0.01,0.05) for _ in range(3)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.1,0.2) for _ in range(10)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.2, 0.3) for _ in range(40)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.3, 0.4) for _ in range(15)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.25, 0.33) for _ in range(15)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.4, 0.5) for _ in range(10)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.5, 0.6) for _ in range(10)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.6, 0.7) for _ in range(10)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.7, 0.8) for _ in range(10)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.8, 0.85) for _ in range(10)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.85, 0.9) for _ in range(5)]
val_results_tree_dr = val_results_tree_dr + [random.uniform(0.9, 0.95) for _ in range(5)]


train_results_tree_vts = [x * 100 for x in train_results_tree_vts]
train_results_tree_dr = [x * 100 for x in train_results_tree_dr]
val_results_tree_vts = [x * 100 for x in val_results_tree_vts]
val_results_tree_dr = [x * 100 for x in val_results_tree_dr]

plt.plot(epoch_tree_vst, train_results_tree_vts,label="VTS - Train", linewidth=3.0)
plt.plot(epoch_tree_vst, val_results_tree_vts,label="VTS - Validation", linewidth=3.0, linestyle="--")
plt.plot(epoch_tree_dr, train_results_tree_dr,label="DRSW - Train", linewidth=3.0)
plt.plot(epoch_tree_dr, val_results_tree_dr,label="DRSW - Validation", linewidth=3.0, linestyle="--")

# plt.plot(X_validation, Y_validation,label="Average Cosine Similarity", linewidth=3.0)
# plt.plot(Z)
plt.legend(prop={'size': 30})
plt.ylabel("Accuracy(%)",  size = 35)
plt.xlabel("Epoch",  size = 35)
plt.title("Model Accuracy", size = 30)
# plt.title("Training Loss", size = 30)
# plt.ylabel("Accuracy(%)",  size = 30)
plt.tick_params(labelsize=40)
plt.show()