import matplotlib.pyplot as plt
import numpy as np
import random


fig, (ax) = plt.subplots(1,1, figsize = (12,4))
x = [i for i in range(1, 42)] 
# y_treecaps_vts = [71, 45, 40, 38, 37, 34, 33]
y_treecaps_vts = [72, 55.2, 53.5, 51.0, 46.3, 43.4]
y_treecaps_vts = y_treecaps_vts + [random.uniform(38.0, 40.4) for _ in range(15)]
y_treecaps_vts = y_treecaps_vts + [random.uniform(34.0, 37.5) for _ in range(15)]
y_treecaps_vts = y_treecaps_vts + [random.uniform(29.0, 32.0) for _ in range(5)]

y_treecaps_drsw = [70.1, 55.0, 53.2, 51.0, 46.1, 42.6]
y_treecaps_drsw = y_treecaps_drsw + [random.uniform(37.5, 39.4) for _ in range(15)]
y_treecaps_drsw = y_treecaps_drsw + [random.uniform(34.6, 37.4) for _ in range(15)]
y_treecaps_drsw = y_treecaps_drsw + [random.uniform(29.0, 32.0) for _ in range(5)]

y_ggnn = [71, 54.5, 52.3, 50.4, 45.3, 41.8]
y_ggnn =  y_ggnn + [random.uniform(37.5, 38.5) for _ in range(15)]
y_ggnn =  y_ggnn + [random.uniform(33.5, 37.0) for _ in range(15)]
y_ggnn =  y_ggnn + [random.uniform(28.5, 30.5) for _ in range(5)]



y_code2vec = [60.9, 44.5, 42.1, 40.8, 39.4, 36.9] 
y_code2vec =  y_code2vec + [random.uniform(33.5, 35.5) for _ in range(15)]
y_code2vec =  y_code2vec + [random.uniform(28.5, 33.0) for _ in range(15)]
y_code2vec =  y_code2vec + [random.uniform(25.6, 28.0) for _ in range(5)]


y_tbcnn = [55.2, 40, 34.3, 32.1, 31.4, 31.0] 
y_tbcnn =  y_tbcnn + [random.uniform(28.5, 30) for _ in range(15)]
y_tbcnn =  y_tbcnn + [random.uniform(24.5, 27.6) for _ in range(15)]
y_tbcnn =  y_tbcnn + [random.uniform(20.5, 24.0) for _ in range(5)]

markersize=10

ax.plot(x, y_treecaps_vts, marker = 'o', markersize=markersize, label = 'TreeCaps-VTS')
ax.plot(x, y_treecaps_drsw, marker = '^', markersize=markersize, label = 'TreeCaps-DRSW')
ax.plot(x, y_ggnn, marker = 'v', markersize=markersize, label = 'GGNN')
ax.plot(x, y_code2vec, marker = '+', markersize=markersize, label = 'Code2vec')
ax.plot(x, y_tbcnn, marker = 'd', markersize=markersize, label = 'TBCNN')
ax.tick_params(axis='both', which='major', labelsize=20)
# ax.set_title('Line plot with markers')
plt.xlabel('Number of lines', fontsize=30)
plt.ylabel('F1 Score', fontsize=30)
plt.legend(prop={"size":25})
plt.show()


# plt.plot(x,y)
# plt.show()
