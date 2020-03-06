import pickle
import numpy as np
import random
from tqdm import *

from keras.preprocessing.sequence import pad_sequences

def gen_samples(trees, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}
    random.shuffle(trees)
    # trees_test = []
    # for i in range(len(trees)):
    #     trees_test.append(trees[3])
    for tree in trees:
    
        nodes = []
        children = []
        label = label_lookup[tree['label']]

        queue = [(tree['tree'], -1)]
        while queue:
            node, parent_ind = queue.pop(0)
            node_ind = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            
            n = str(node['node'])
            # look_up_vector = vector_lookup[n]
            # nodes.append(vectors[int(n)])
            nodes.append(int(n))

        if len(nodes) > 5000 :
            if len(children) < 700:
                yield (nodes, children, label)
        else:
            yield (nodes, children, label)


def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels = [], [], []
    samples = 0
    for n, c, l in gen:
        nodes.append(n)
        children.append(c)
        labels.append(l)
        samples += 1
        if samples >= batch_size:
            # nodes_indicator = _produce_mask_vector(nodes)
            nodes, children, labels = _pad_batch(nodes, children, labels)

            yield nodes, children, labels
            nodes, children, labels = [], [], []
            samples = 0

    # if nodes:
    #     yield _pad_batch(nodes, children, labels)


def _pad_batch(nodes, children, labels):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    # for i in range(1):
    #     children[0].append([])
    max_children = max([len(x) for x in children])
    # feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    # max_nodes = max_nodes + 1
    # nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    nodes = [n + [0] * (max_nodes - len(n)) for n in nodes]
   
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return np.asarray(nodes), np.asarray(children), np.asarray(labels)


def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]

def _produce_mask_vector(nodes):

    masks = []
   
    for n in nodes:        
        mask = [1 for i in range(len(n))]
        masks.append(mask)

    padded_inputs = pad_sequences(masks, padding='post')
    return padded_inputs
