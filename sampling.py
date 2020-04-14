import pickle
import numpy as np
import random
from tqdm import *
import config
from keras.preprocessing.sequence import pad_sequences

def gen_samples(trees, label_size, vector_lookup, batch_size):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors

    # trees_test = []
    # for i in range(len(trees)):
    #     trees_test.append(trees[3])
    elements = []
    samples = 0
    for tree in trees:
        size = tree["size"]

        if size < config.FILE_SIZE_THRESHOLD:
            elements.append(tree)
            samples += 1

        if samples >= batch_size:
            # print("Num samples : " + str(samples))
            batch_nodes, batch_children, batch_labels = [], [], []
            for element in elements:
                print(element["label"])
                nodes, children, labels = extract_training_data(element, label_size)
                batch_nodes.append(nodes)
                batch_children.append(children)
                batch_labels.append(labels)

            # batch_nodes = pad_sequences(batch_nodes, padding='post')
            # batch_children = pad_sequences(batch_children, padding='post')
            batch_nodes, batch_children = _pad_batch(batch_nodes, batch_children)
            # print(len(batch_nodes[0]))
            # print(len(batch_nodes[1]))
            # print(len(batch_nodes[2]))
            # print(len(batch_nodes[3]))
            yield batch_nodes, batch_children, batch_labels
            elements = []
            samples = 0

def extract_training_data(tree, label_size):
    nodes = []
    children = []
    label_one_hot = _onehot(tree["label"], label_size)
    
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

    return nodes, children, label_one_hot


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
            print(labels)
            yield nodes, children, labels
            nodes, children, labels = [], [], []
            samples = 0

    # if nodes:
    #     yield _pad_batch(nodes, children, labels)


def _pad_batch(nodes, children):
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
    return np.asarray(nodes), np.asarray(children)
    # return nodes, children, labels

def _onehot(label, total):
    return [1.0 if j == (label-1) else 0.0 for j in range(total)]

def _produce_mask_vector(nodes):

    masks = []
   
    for n in nodes:        
        mask = [1 for i in range(len(n))]
        masks.append(mask)

    padded_inputs = pad_sequences(masks, padding='post')
    return padded_inputs
