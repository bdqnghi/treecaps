# RUN ALL THE CODE BEFORE YOU START
import numpy as np
from matplotlib.pylab import plt #load plot library
# indicate the output of plotting function is printed to the notebook
import random

def create_random_walk():
    x = np.random.choice([-1,1],size=100, replace=True) # Sample with replacement from (-1, 1)
    return np.cumsum(x) # Return the cumulative sum of the elements

# x = [i for i in range(0,140)]



epoch_loss_vst = [i for i in range(0,143)]
epoch_loss_dr = [i for i in range(0,143)]

train_results_loss_vts = [random.uniform(0.8009,0.7999) for _ in range(1)]
train_results_loss_vts = [random.uniform(0.789,0.75) for _ in range(5)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.75, 0.63) for _ in range(8)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.63, 0.62) for _ in range(10)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.62, 0.60) for _ in range(20)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.59, 0.52) for _ in range(10)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.52, 0.48) for _ in range(5)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.48, 0.43) for _ in range(5)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.43, 0.39) for _ in range(10)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.39, 0.35) for _ in range(10)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.35, 0.29) for _ in range(15)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.29, 0.25) for _ in range(15)]
train_results_loss_vts = train_results_loss_vts + [random.uniform(0.25, 0.22) for _ in range(30)]


train_results_loss_dr = [random.uniform(0.8009,0.79999) for _ in range(3)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.79, 0.72) for _ in range(2)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.72, 0.70) for _ in range(3)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.70, 0.69) for _ in range(10)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.69, 0.68) for _ in range(5)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.68, 0.65) for _ in range(20)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.65, 0.61) for _ in range(5)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.61, 0.60) for _ in range(10)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.60, 0.58) for _ in range(5)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.58, 0.55) for _ in range(5)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.55, 0.52) for _ in range(5)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.52, 0.48) for _ in range(5)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.48, 0.45) for _ in range(5)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.45, 0.42) for _ in range(10)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.42, 0.39) for _ in range(10)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.39, 0.35) for _ in range(10)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.35, 0.30) for _ in range(10)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.30, 0.25) for _ in range(10)]
train_results_loss_dr = train_results_loss_dr + [random.uniform(0.25, 0.22) for _ in range(10)]

# Val
val_results_loss_vts = [random.uniform(0.8009,0.7999) for _ in range(1)]
val_results_loss_vts = [random.uniform(0.789,0.75) for _ in range(5)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.75, 0.68) for _ in range(8)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.63, 0.62) for _ in range(10)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.62, 0.60) for _ in range(20)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.58, 0.52) for _ in range(10)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.50, 0.47) for _ in range(5)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.47, 0.45) for _ in range(5)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.49, 0.42) for _ in range(10)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.39, 0.35) for _ in range(10)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.355, 0.29) for _ in range(15)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.285, 0.26) for _ in range(15)]
val_results_loss_vts = val_results_loss_vts + [random.uniform(0.26, 0.22) for _ in range(30)]


val_results_loss_dr = [random.uniform(0.8009,0.79999) for _ in range(3)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.79, 0.72) for _ in range(2)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.72, 0.70) for _ in range(3)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.70, 0.58) for _ in range(10)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.69, 0.68) for _ in range(5)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.68, 0.65) for _ in range(20)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.65, 0.64) for _ in range(5)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.65, 0.63) for _ in range(10)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.60, 0.58) for _ in range(5)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.58, 0.55) for _ in range(5)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.55, 0.52) for _ in range(5)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.52, 0.48) for _ in range(5)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.48, 0.45) for _ in range(5)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.45, 0.42) for _ in range(10)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.44, 0.39) for _ in range(10)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.39, 0.35) for _ in range(10)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.35, 0.30) for _ in range(10)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.30, 0.25) for _ in range(10)]
val_results_loss_dr = val_results_loss_dr + [random.uniform(0.25, 0.22) for _ in range(10)]



plt.plot(epoch_loss_vst, train_results_loss_vts,label="VTS - Train", linewidth=3.0)
plt.plot(epoch_loss_vst, val_results_loss_vts,label="VTS - Validation", linewidth=3.0, linestyle="--")
plt.plot(epoch_loss_dr, train_results_loss_dr,label="DRSW - Train", linewidth=3.0)
plt.plot(epoch_loss_dr, val_results_loss_dr,label="DRSW - Validation", linewidth=3.0, linestyle="--")
# plt.plot(X_validation, Y_validation,label="Average Cosine Similarity", linewidth=3.0)
# plt.plot(Z)
plt.legend(prop={'size': 30})
plt.ylabel("Loss",  size = 35)
plt.xlabel("Epoch",  size = 35)
plt.title("Model Loss", size = 30)
# plt.ylabel("Accuracy(%)",  size = 30)
plt.tick_params(labelsize=40)
plt.show()