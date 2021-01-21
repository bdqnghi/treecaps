import pickle
import numpy as np
import random
from tqdm import *

from keras.preprocessing.sequence import pad_sequences

def gen_samples(trees, labels):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}
    random.shuffle(trees)
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
            # print(n)
            nodes.append(int(n))

    
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
            yield _pad_batch(nodes, children, labels)
            nodes, children, labels = [], [], []
            samples = 0

def _pad_batch(nodes, children, labels):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [0] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children, labels


def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]


def _pad_batch_siamese_2_side(batch_left_nodes, batch_left_children, batch_right_nodes, batch_right_children, labels):
    max_left_nodes = max([len(x) for x in batch_left_nodes])
    max_right_nodes = max([len(x) for x in batch_right_nodes])

    max_left_children = max([len(x) for x in batch_left_children])
    max_right_children = max([len(x) for x in batch_right_children])

    max_nodes = max(max_left_nodes, max_right_nodes)
    max_children = max(max_left_children, max_right_children)

    left_masks, right_masks = _produce_mask_vector_2_side(batch_left_nodes, batch_right_nodes, max_left_nodes, max_right_nodes)
    batch_left_nodes, batch_left_children = _pad_batch_siamese(batch_left_nodes, batch_left_children, max_nodes, max_children)
    batch_right_nodes, batch_right_children = _pad_batch_siamese(batch_right_nodes, batch_right_children, max_nodes, max_children)

    return (batch_left_nodes, batch_left_children,left_masks), (batch_right_nodes, batch_right_children,right_masks), labels

def _produce_mask_vector(nodes, max_nodes):

    masks = []
    mask = [1 for i in range(max_nodes)]
    for n in nodes:        
        masks.append(mask)

    return masks
