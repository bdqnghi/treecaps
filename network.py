import math
import tensorflow as tf
from utils import reduce_sum
from utils import softmax
from utils import get_shape
import numpy as np
import config
from config import BATCH_SIZE

stddev = 0.01
batch_size = BATCH_SIZE
iter_routing = 3
regular_sc = 1e-7
epsilon = 1e-11
m_plus = 0.9
m_minus = 0.1

def squash(vector):
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

def dynamic_routing(shape, input, num_outputs=10, num_dims=16):
    """The Dynamic Routing Algorithm proposed by Sabour et al."""
    # input shape: (12, 1280, 1, 8, 1)
    input_shape = shape
    # (1, 1280, 80, 8, 1)
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))
    
    delta_IJ = tf.zeros([input_shape[0], input_shape[1], num_outputs, 1, 1], dtype=tf.dtypes.float32)

    # input shape: (12, 1280, 80, 8, 1)
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])

    # (12, 1280, 1, 8, 1)
    u_hat = reduce_sum(W * input, axis=3, keepdims=True)

    # (12, 1280, 1, 8, 1)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])

    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            gamma_IJ = softmax(delta_IJ, axis=2)

            if r_iter == iter_routing - 1:
                s_J = tf.multiply(gamma_IJ, u_hat)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(gamma_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                delta_IJ += u_produce_v

    # (batch_size, 1, num_outputs, num_dims, 1) = (12, 1, 10, 8, 1)

    return(v_J)
 

def dynamic_routing_code_caps(w, b, primary_variable_caps, output_size=80, output_dimension=16, input_dimension=8, num_set_of_code_caps=2):
        # input_dimension: dimension of input capsule
        # output_dimension: dimension of output capsule
        # output_size: number of capsules

      
        w_dynamic_routing = w 
        b_dynamic_routing = b
        
        # tile with max_node = (1, output_size x max_node, 640, 2, 1)
        # (1, output_size x max_node, output_dimension x output_size x num_set_of_code_caps, num_conv, 1)
        w_dynamic_routing = tf.tile(w_dynamic_routing, [1, tf.shape(primary_variable_caps)[1], 1, 1, 1])

        # (batch_size, output_size x max_node, 1, num_conv, 1)
        primary_variable_caps_reshaped = tf.reshape(primary_variable_caps, shape=[batch_size, -1, 1, input_dimension, 1])
        
        # (batch_size, output_size x max_node, output_dimension * output_size * num_set_of_code_caps, num_conv, 1)
        primary_variable_caps_tiled = tf.tile(primary_variable_caps_reshaped, [1, 1, output_dimension * output_size* num_set_of_code_caps, 1, 1])

        # (3, output_size x max_node, output_dimension x output_size x num_set_of_code_caps, num_conv, 1)
        u_hat = w_dynamic_routing * primary_variable_caps_tiled
        # (3, output_size x max_node, output_dimension x output_size x num_set_of_code_caps, 1, 1)
        u_hat = tf.reduce_sum(u_hat, axis=3, keep_dims=True)
        
        # (3, output_size x max_node, num_set_of_code_caps x output_size, output_dimension, 1)
        u_hat = tf.reshape(u_hat, shape=[-1, tf.shape(primary_variable_caps_reshaped)[1], num_set_of_code_caps * output_size, output_dimension, 1])
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
        
        # # (3, output_size x max_node, num_set_of_code_caps x output_size, 1, 1)
        delta_IJ = tf.zeros([batch_size, tf.shape(primary_variable_caps_reshaped)[1], num_set_of_code_caps * output_size, 1, 1], dtype=tf.dtypes.float32)
        for r_iter in range(iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                gamma_IJ = tf.nn.softmax(delta_IJ, axis=2)

                if r_iter == iter_routing - 1:
                    # (3, output_size x max_node, num_set_of_code_caps x output_size, output_dimension, 1)
                    s_J = tf.multiply(gamma_IJ, u_hat)
                    # (3, 1, 2 x 80, 8, 1)
                    s_J_1 = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                    v_J = squash(s_J_1)
                elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                    # (3, output_size x max_node, num_set_of_code_caps x output_size, output_dimension, 1)
                    s_J = tf.multiply(gamma_IJ, u_hat_stopped)
                    # (3, 1, 2 x 80, 8, 1)
                    s_J_1 = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                    # (3, 1, 2 x 80, 8, 1)
                    v_J = squash(s_J_1)
                    # (3, output_size x max_node, num_set_of_code_caps x output_size, output_dimension, 1)
                    v_J_tiled = tf.tile(v_J, [1,tf.shape(primary_variable_caps_reshaped)[1], 1, 1, 1])
                    # (3, output_size x max_node, num_set_of_code_caps x output_size, 1, 1)
                    u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                    # (3, output_size x max_node, num_set_of_code_caps x output_size, 1, 1)
                    delta_IJ += u_produce_v
        
        return v_J


def dynamic_routing_class_caps(w, b, primary_variable_caps, output_size=80, output_dimension=16, input_dimension=8):
    # input_dimension: dimension of input capsule
    # output_dimension: dimension of output capsule
    # output_size: number of capsules

    # (1, input_size, output_dimension x output_size, input_dimension, 1)
    # (1, 50, 8 * 80, 2, 1)
    # (1, 50, 640, 2, 1)
    # w_dynamic_routing = self.placeholders["w_dynamic_routing_code_caps"]
    w_dynamic_routing = w 
    # (1, 1, 80, 8, 1)
    # b_dynamic_routing = self.placeholders["b_dynamic_routing_code_caps"]
    b_dynamic_routing = b
    
    # tile with max_node = (1, output_size x max_node, 640, 2, 1)
    # (1, output_size x max_node, output_dimension x output_size, 2, 1)
    # w_dynamic_routing = tf.tile(w_dynamic_routing, [1, tf.shape(primary_variable_caps)[1], 1, 1, 1])

   
    # (batch_size, output_size x max_node, 1, num_conv, 1)
    primary_variable_caps_reshaped = tf.reshape(primary_variable_caps, shape=[batch_size, -1, 1, input_dimension, 1])
    
    # (batch_size, output_size x max_node, output_dimension * output_size, num_conv, 1)
    # (3, 80*48, 16*80, 2, 1)
    primary_variable_caps_tiled = tf.tile(primary_variable_caps_reshaped, [1, 1, output_dimension * output_size, 1, 1])

    # (3, output_size x max_node, output_dimension x output_size, num_conv, 1)
    u_hat = w_dynamic_routing * primary_variable_caps_tiled
    # (3, output_size x max_node, output_dimension x output_size, 1, 1)
    u_hat = tf.reduce_sum(u_hat, axis=3, keep_dims=True)
    
    # (3, output_size x max_node, output_size, output_dimension, 1)
    u_hat = tf.reshape(u_hat, shape=[-1, tf.shape(primary_variable_caps_reshaped)[1], output_size, output_dimension, 1])

    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # (3, output_size x max_node, 80, 1, 1)

    delta_IJ = tf.zeros([batch_size, tf.shape(primary_variable_caps_reshaped)[1], output_size, 1, 1], dtype=tf.dtypes.float32)
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            gamma_IJ = tf.nn.softmax(delta_IJ, axis=2)

            if r_iter == iter_routing - 1:
                # (3, output_size x max_node, output_size, output_dimension, 1)
                s_J = tf.multiply(gamma_IJ, u_hat)
                # (3, 1, 80, 8, 1)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                v_J = squash(s_J)
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                # (3, output_size x max_node, output_size, output_dimension, 1)
                s_J = tf.multiply(gamma_IJ, u_hat_stopped)
                # (3, 1, 80, 8, 1)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                # (3, 1, 80, 8, 1)
                v_J = squash(s_J)
                # (3, output_size x max_node, output_size, output_dimension, 1)
                v_J_tiled = tf.tile(v_J, [1,tf.shape(primary_variable_caps_reshaped)[1], 1, 1, 1])
                # (3, output_size x max_node, output_size, 1, 1)
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # (3, output_size x max_node, output_size, 1, 1)
                delta_IJ += u_produce_v

    return v_J

def init_net_treecaps(feature_size, label_size, embedding_lookup):
    """Initialize an empty TreeCaps network."""
    top_a = config.TOP_A
    # top_b = config.TOP_B
    
    num_conv = config.NUM_CONV
    output_size = config.OUTPUT_SIZE
    code_caps_num_caps = config.CODE_CAPS_NUM_CAPS
    code_caps_output_dimension = config.CODE_CAPS_OUTPUT_DIMENSION
    class_caps_num_caps = label_size
    class_caps_output_dimension = config.CLASS_CAPS_OUTPUT_DIMENSION
    num_set_of_code_caps = config.NUM_SET_OF_CODE_CAPS

    with tf.name_scope('inputs'):
        # nodes = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
        nodes = tf.placeholder(tf.int32, shape=(None, None), name='tree')
        children = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
        node_embeddings_lookup = tf.Variable(tf.contrib.layers.xavier_initializer()([len(embedding_lookup.keys()), feature_size]), name='node_type_embeddings')
        labels = tf.placeholder(tf.float32, (None, label_size,))

        node_embeddings_lookup = tf.Variable(tf.contrib.layers.xavier_initializer()([len(embedding_lookup.keys()), feature_size]), name='node_type_embeddings')
        # nodes_indicator = tf.placeholder(tf.float32, shape=(None, None), name='nodes_indicator')
        shape_of_weight_dynamic_routing_code_caps = [1, output_size, code_caps_output_dimension * code_caps_num_caps * num_set_of_code_caps,  num_conv, 1]
        shape_of_bias_dynamic_routing_code_caps = [1, 1, num_set_of_code_caps * code_caps_num_caps, code_caps_output_dimension, 1]

        w_dynamic_routing_code_caps = tf.Variable(tf.contrib.layers.xavier_initializer()(shape_of_weight_dynamic_routing_code_caps), name='w_dynamic_routing_code_caps')
        b_dynamic_routing_code_caps = tf.Variable(tf.zeros(shape_of_bias_dynamic_routing_code_caps), name='b_dynamic_routing_code_caps')

        shape_of_weight_dynamic_routing_class_caps = [1, code_caps_num_caps * num_set_of_code_caps, class_caps_output_dimension * class_caps_num_caps, code_caps_output_dimension, 1]
        shape_of_bias_dynamic_routing_class_caps = [1, 1, class_caps_num_caps, class_caps_output_dimension, 1]

        w_dynamic_routing_class_caps = tf.Variable(tf.contrib.layers.xavier_initializer()(shape_of_weight_dynamic_routing_class_caps), name='w_dynamic_routing_class_caps')
        b_dynamic_routing_class_caps = tf.Variable(tf.zeros(shape_of_bias_dynamic_routing_class_caps), name='b_dynamic_routing_class_caps')

    with tf.name_scope('network'):  
        node_embeddings = compute_parent_node_types_tensor(nodes, node_embeddings_lookup)

        # conv_output = conv_layer(num_conv, output_size, node_embeddings, children, feature_size)

        # attention_scores = aggregation_layer(conv_output, output_size, 1)

        # conv_output = attention_scores * conv_output
        """The Primary Variable Capsule Layer."""
        # (12, max_tree_size, num_conv, 128)
        # primary_variable_caps = tf.expand_dims(conv_output, axis=2)
        primary_variable_caps = primary_variable_capsule_layer(num_conv, output_size, node_embeddings, children, feature_size)
        # primary_variable_capsules = tf.reshape(conv_output,shape=(1,-1,output_size,num_conv))

        code_caps = dynamic_routing_code_caps(w_dynamic_routing_code_caps, b_dynamic_routing_code_caps, 
                                            primary_variable_caps, output_size=code_caps_num_caps, output_dimension=code_caps_output_dimension, 
                                            input_dimension=num_conv, num_set_of_code_caps=num_set_of_code_caps)
        # batch size = 1: (12, 80, 8, 1)
        code_caps = tf.squeeze(code_caps, axis=1)

        
        class_caps = dynamic_routing_class_caps(w_dynamic_routing_class_caps, 
                                b_dynamic_routing_class_caps, code_caps, 
                                output_size=class_caps_num_caps, output_dimension=class_caps_output_dimension, 
                                input_dimension=code_caps_output_dimension)
        # batch size = 12: (12, num_class_cap, 50, 1)
        class_caps = tf.squeeze(class_caps, axis=1)
        
     
        # (12, 10, 1, 1)
        # # """Obtaining the classification output."""
        v_length = tf.sqrt(reduce_sum(tf.square(class_caps),axis=2, keepdims=True) + 1e-9)
        out = tf.reshape(v_length,(-1,label_size))

        loss = loss_layer(out, labels)

    return nodes, children, out, loss, labels

def aggregation_layer(conv, output_size, distributed_function):
    # conv is (batch_size, max_tree_size, output_size)
    with tf.name_scope("global_attention"):
        initializer = tf.contrib.layers.xavier_initializer()
        w_attention = tf.Variable(initializer([output_size,1]), name='w_attention')

        batch_size = tf.shape(conv)[0]
        max_tree_size = tf.shape(conv)[1]

        flat_conv = tf.reshape(conv, [-1, output_size])
        aggregated_vector = tf.matmul(flat_conv, w_attention)

        attention_score = tf.reshape(aggregated_vector, [-1, max_tree_size, 1])

        if distributed_function == 0:
            attention_weights = tf.nn.softmax(attention_score, dim=1)
        if distributed_function == 1:
            attention_weights = tf.nn.sigmoid(attention_score)

        return attention_weights

def compute_parent_node_types_tensor(nodes, node_embeddings_lookup):
    node_embeddings =  tf.nn.embedding_lookup(node_embeddings_lookup,nodes)
    return node_embeddings

def nn_attention_layer(inputs,batch_size,mask,name,emb_size,channel_num):
    """
    :param inputs: (batch, N, C, d)
    :param batch_size:
    :param mask: (batch, N, 1)
    :param name:
    :param emb_size: int(d)
    :param channel_num: int(C)
    :return: (batch, N, C, d)
    """

    N = tf.shape(inputs)[1]
    with tf.variable_scope(name) as scope:
        inputs_ = tf.reshape(inputs, shape=[batch_size * N, emb_size * channel_num])  # (?*N, C*d)
        atten = tf.layers.dense(inputs_, units=int(emb_size * channel_num / 16), activation=tf.nn.tanh)  # (?*N, C*d/16)
        atten = tf.layers.dense(atten, units=channel_num, activation=None)  # (?*N, C)
        atten = tf.reshape(atten, shape=[batch_size, N, channel_num, 1])  # (?, N, C, 1)
        atten = mask_softmax(atten, mask, dim=1)  # (batch, N, C, 1)
        
        # (batch, N, 8, 128) x (batch, N, 8, 1)
        input_scaled = tf.multiply(inputs, atten)  # (batch, N, C, 1)
        num_nodes = tf.expand_dims(tf.reduce_sum(mask, axis=1, keep_dims=True), axis=-1)
        input_scaled = input_scaled * num_nodes

    return input_scaled

def mask_softmax(inputs,mask,dim):
    """
    :param inputs: (batch, N, C, 1)
    :param mask: (batch, N, 1)
    :param dim: does softmax along which axis
    :return: normalized attention (batch, N, C, 1)
    """
    
    with tf.variable_scope('bulid_mask') as scope:
        e_inputs = tf.exp(inputs) + epsilon # (batch, N, C, 1)
        mask = mask[:,:,:,tf.newaxis]  # (batch, N, 1, 1)
        mask = tf.tile(mask,multiples=[1,1,tf.shape(e_inputs)[2],1])  # (batch, N, C, 1)
        masked_e_inputs = tf.multiply(e_inputs,mask)  # (batch, N, C, 1)
        sum_col = tf.reduce_sum(masked_e_inputs,axis=dim,keep_dims=True) +epsilon  # (batch, 1, C, 1)
        result = tf.div(masked_e_inputs,sum_col)  # (batch, N, C, 1)
    return result


def primary_variable_capsule_layer(num_conv, output_size, nodes, children, feature_size):
    """The proposed Primary Variable Capsule Layer."""
    
    with tf.name_scope('primary_variable_capsule_layer'):
        nodes = [
            tf.expand_dims(conv_node(nodes, children, feature_size, output_size),axis=-1)
            for _ in range(num_conv)
        ]    
        conv_output = tf.concat(nodes, axis=-1)
        return conv_output


def conv_layer(num_conv, output_size, nodes, children, feature_size):
    """Creates a convolution layer with num_conv convolutions merged together at
    the output. Final output will be a tensor with shape
    [batch_size, num_nodes, output_size * num_conv]"""

    with tf.name_scope('conv_layer'):
        nodes = [
            conv_node(nodes, children, feature_size, output_size)
            for _ in range(num_conv)
        ]     
        # return tf.concat(nodes, axis=-1)
        return tf.concat(nodes, axis=2)


# def conv_layer(num_conv, output_size, nodes, children, feature_size):
#     """Creates a convolution layer with num_conv convolutions merged together at
#     the output. Final output will be a tensor with shape
#     [batch_size, num_nodes, output_size * num_conv]"""

#     with tf.name_scope('conv_layer'):
#         nodes = [
#             tf.expand_dims(conv_node(nodes, children, feature_size, output_size),axis=-1)
#             for _ in range(num_conv)
#         ]     
#         return tf.concat(nodes, axis=-1)

def conv_node(nodes, children, feature_size, output_size):
    """Perform convolutions over every batch sample."""
    with tf.name_scope('conv_node'):
        std = 1.0 / math.sqrt(feature_size)
        w_t, w_l, w_r = (
            tf.Variable(tf.random.truncated_normal([feature_size, output_size], stddev=std), name='Wt'),
            tf.Variable(tf.random.truncated_normal([feature_size, output_size], stddev=std), name='Wl'),
            tf.Variable(tf.random.truncated_normal([feature_size, output_size], stddev=std), name='Wr'),
        )
        init = tf.random.truncated_normal([output_size,], stddev=math.sqrt(2.0/feature_size))

        b_conv = tf.Variable(init, name='b_conv')

        return conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv)



def conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv):
    """Convolve a batch of nodes and children.
    Lots of high dimensional tensors in this function. Intuitively it makes
    more sense if we did this work with while loops, but computationally this
    is more efficient. Don't try to wrap your head around all the tensor dot
    products, just follow the trail of dimensions.
    """
    with tf.name_scope('conv_step'):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        # children is shape (batch_size x max_tree_size x max_children)

        with tf.name_scope('trees'):
            # children_vectors will have shape
            # (batch_size x max_tree_size x max_children x feature_size)
            children_vectors = children_tensor(nodes, children, feature_size)

            # add a 4th dimension to the nodes tensor
            nodes = tf.expand_dims(nodes, axis=2)
            # tree_tensor is shape
            # (batch_size x max_tree_size x max_children + 1 x feature_size)
            tree_tensor = tf.concat([nodes, children_vectors], axis=2, name='trees')

        with tf.name_scope('coefficients'):
            # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
            c_t = eta_t(children)
            c_r = eta_r(children, c_t)
            c_l = eta_l(children, c_t, c_r)

            # concatenate the position coefficients into a tensor
            # (batch_size x max_tree_size x max_children + 1 x 3)
            coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

        with tf.name_scope('weights'):
            # stack weight matrices on top to make a weight tensor
            # (3, feature_size, output_size)
            weights = tf.stack([w_t, w_r, w_l], axis=0)

        with tf.name_scope('combine'):
            batch_size = tf.shape(children)[0]
            max_tree_size = tf.shape(children)[1]
            max_children = tf.shape(children)[2]

            # reshape for matrix multiplication
            x = batch_size * max_tree_size
            y = max_children + 1
            result = tf.reshape(tree_tensor, (x, y, feature_size))
            coef = tf.reshape(coef, (x, y, 3))
            result = tf.matmul(result, coef, transpose_a=True)
            result = tf.reshape(result, (batch_size, max_tree_size, 3, feature_size))

            # output is (batch_size, max_tree_size, output_size)
            result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

            # output is (batch_size, max_tree_size, output_size)
            return tf.nn.tanh(result + b_conv)



def children_tensor(nodes, children, feature_size):
    """Build the children tensor from the input nodes and child lookup."""
    with tf.name_scope('children_tensor'):
        max_children = tf.shape(children)[2]
        batch_size = tf.shape(nodes)[0]
        num_nodes = tf.shape(nodes)[1]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, feature_size))
        # vector_lookup is (batch_size x num_nodes x feature_size)
        # vector_lookup = nodes
        vector_lookup = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children = tf.expand_dims(children, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children = tf.concat([batch_indices, children], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x feature_size)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children, name='children')

def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    with tf.name_scope('coef_t'):
        # children is shape (batch_size x max_tree_size x max_children)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]
        # eta_t is shape (batch_size x max_tree_size x max_children + 1)
        return tf.tile(tf.expand_dims(tf.concat(
            [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
            axis=1), axis=0,
        ), [batch_size, 1, 1], name='coef_t')

def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belogs to the 'right'"""
    with tf.name_scope('coef_r'):
        # children is shape (batch_size x max_tree_size x max_children)
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]

        # num_siblings is shape (batch_size x max_tree_size x 1)
        num_siblings = tf.cast(
            tf.count_nonzero(children, axis=2, keep_dims=True),
            dtype=tf.float32
        )
        # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
        num_siblings = tf.tile(
            num_siblings, [1, 1, max_children + 1], name='num_siblings'
        )
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2, name='mask'
        )

        # child indices for every tree (batch_size x max_tree_size x max_children + 1)
        child_indices = tf.multiply(tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                    axis=0
                ),
                axis=0
            ),
            [batch_size, max_tree_size, 1]
        ), mask, name='child_indices')

        # weights for every tree node in the case that num_siblings = 0
        # shape is (batch_size x max_tree_size x max_children + 1)
        singles = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.fill((batch_size, max_tree_size, 1), 0.5),
             tf.zeros((batch_size, max_tree_size, max_children - 1))],
            axis=2, name='singles')

        # eta_r is shape (batch_size x max_tree_size x max_children + 1)
        return tf.where(
            tf.equal(num_siblings, 1.0),
            # avoid division by 0 when num_siblings == 1
            singles,
            # the normal case where num_siblings != 1
            tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
            name='coef_r'
        )

def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    with tf.name_scope('coef_l'):
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
                tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2,
            name='mask'
        )

        # eta_l is shape (batch_size x max_tree_size x max_children + 1)
        return tf.multiply(
            tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
        )

def pooling_layer(nodes):
    """Creates a max dynamic pooling layer from the nodes."""
    with tf.name_scope("pooling"):
        pooled = tf.reduce_max(nodes, axis=1)
        return pooled


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def hidden_layer(pooled, input_size, output_size):
    """Create a hidden feedforward layer."""
    with tf.name_scope("hidden"):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_size, output_size], stddev=1.0 / math.sqrt(input_size)
            ),
            name='weights'
        )

        init = tf.truncated_normal([output_size,], stddev=math.sqrt(2.0/input_size))
        #init = tf.zeros([output_size,])
        biases = tf.Variable(init, name='biases')

        return lrelu(tf.matmul(pooled, weights) + biases, 0.01)

def loss_layer(logits_node, labels):
    """Create a loss layer for training."""

    with tf.name_scope('loss_layer'):
        max_l = tf.square(tf.maximum(0., 0.9 - logits_node))
        max_r = tf.square(tf.maximum(0., logits_node - 0.1))
        T_c = labels
        L_c = T_c * max_l + 0.5 * (1 - T_c) * max_r
        
        loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        return loss

def out_layer(logits_node):
    """Apply softmax to the output layer."""
    with tf.name_scope('output'):
        return tf.nn.softmax(logits_node)