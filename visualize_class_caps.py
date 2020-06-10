import gensim
import os
import codecs
from sklearn.manifold import TSNE
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.getcwd()


considers_label = [1, 2, 3, 4]
def load_embeddings(file_name):
    with codecs.open(file_name,"r","utf-8") as f:
        
        data = f.readlines()
        consider_lines = []
        for line in data:
            line = line.strip()
            splits = line.split(",", 1)
            if int(splits[0]) in considers_label:
                consider_lines.append(line)

        vocabulary, wv = zip(*[line.strip().split(",",2) for line in consider_lines])
    wv = np.loadtxt(wv)
    return wv, vocabulary


path = "analysis/tree_caps_pc_output_size_30-num_conv_8-n_class_104-code_caps_num_caps_10-code_caps_o_dimension_20-class_caps_o_dimension_30/class_caps.txt"

wv,vocabulary = load_embeddings(path)

# print(channel)
# print(vocabulary)
# print(wv)
# print(vocabulary)

tsne = TSNE(n_components=2, random_state=0)

np.set_printoptions(suppress=True)
Y = tsne.fit_transform(wv)

plt.scatter(Y[:,0],Y[:,1])
print("Annotating....")
for label, x, y in zip(vocabulary,Y[:,0],Y[:,1]):
    print("----------------")
    color = "blue"
    if int(label) == 1:
        color = "red"
    if int(label) == 2:
        color = "blue"
    if int(label) == 3:
        color = "green"
    if int(label) == 4:
        color = "orange"

    plt.annotate(label,xy=(x,y),xytext=(0,0),textcoords="offset points", color=color)

plt.show()
