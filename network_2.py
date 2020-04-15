import math
import tensorflow as tf
from utils import reduce_sum
from utils import softmax
from utils import get_shape
import numpy as np
import config

stddev = 0.01
iter_routing = 3
regular_sc = 1e-7
epsilon = 1e-11
m_plus = 0.9
m_minus = 0.1


class TreeCapsModel():
    def __init__(self, opt):
        self.num_conv = opt.num_conv
        self.output_size = opt.output_size
        self.code_caps_num_caps = opt.code_caps_num_caps
        self.code_caps_output_dimension = opt.code_caps_output_dimension
        self.class_caps_num_caps = opt.label_size
        self.class_caps_output_dimension = opt.class_caps_output_dimension
        self.node_type_lookup = opt.node_type_lookup
        self.label_size = opt.label_size
        self.node_type_dim = opt.node_type_dim
        self.batch_size = opt.batch_size

        self.node_dim = self.node_type_dim

        self.placeholders = {}
        self.weights = {}
        self.init_net_treecaps()
        self.feed_forward()

    def init_net_treecaps(self):
        """Initialize parameters"""
        with tf.name_scope('inputs'):
            # nodes = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
            self.placeholders["node_types"] = tf.placeholder(tf.int32, shape=(None, None), name='node_types')
            self.placeholders["children_indices"] = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
            self.placeholders["node_type_embedding_lookup"] = tf.Variable(tf.contrib.layers.xavier_initializer()([len(self.node_type_lookup.keys()), self.node_dim]), name='node_type_embeddings')
            self.placeholders["labels"] = tf.placeholder(tf.float32, (None, self.label_size,))

            self.weights["node_type_embedding_lookup"] = tf.Variable(tf.contrib.layers.xavier_initializer()([len(self.node_type_lookup.keys()), self.node_dim]), name='node_type_embeddings')
            # nodes_indicator = tf.placeholder(tf.float32, shape=(None, None), name='nodes_indicator')
            shape_of_weight_dynamic_routing_code_caps = [1, 1, self.code_caps_output_dimension * self.code_caps_num_caps,  self.num_conv, 1]
            shape_of_bias_dynamic_routing_code_caps = [1, 1, self.code_caps_num_caps, self.code_caps_output_dimension, 1]

            self.weights["w_dynamic_routing_code_caps"] = tf.Variable(tf.contrib.layers.xavier_initializer()(shape_of_weight_dynamic_routing_code_caps), name='w_dynamic_routing_code_caps')
            self.weights["b_dynamic_routing_code_caps"] = tf.Variable(tf.zeros(shape_of_bias_dynamic_routing_code_caps), name='b_dynamic_routing_code_caps')

            shape_of_weight_dynamic_routing_class_caps = [1, self.code_caps_num_caps, self.class_caps_output_dimension * self.class_caps_num_caps, self.code_caps_output_dimension, 1]
            shape_of_bias_dynamic_routing_class_caps = [1, 1, self.class_caps_num_caps, self.class_caps_output_dimension, 1]

            self.weights["w_dynamic_routing_class_caps"] = tf.Variable(tf.contrib.layers.xavier_initializer()(shape_of_weight_dynamic_routing_class_caps), name='w_dynamic_routing_class_caps')
            self.weights["b_dynamic_routing_class_caps"] = tf.Variable(tf.zeros(shape_of_bias_dynamic_routing_class_caps), name='b_dynamic_routing_class_caps')

            for i in range(self.num_conv):
                self.weights["w_t_" + str(i)] = tf.Variable(tf.contrib.layers.xavier_initializer()([self.node_dim, self.output_size]), name='w_t_' + str(i))
                self.weights["w_l_" + str(i)] = tf.Variable(tf.contrib.layers.xavier_initializer()([self.node_dim, self.output_size]), name='w_l_' + str(i))
                self.weights["w_r_" + str(i)] = tf.Variable(tf.contrib.layers.xavier_initializer()([self.node_dim, self.output_size]), name='w_r_' + str(i))
                self.weights["b_conv_" + str(i)] = tf.Variable(tf.zeros([self.output_size,]),name='b_conv_' + str(i))

    def feed_forward(self):
        with tf.name_scope('network'):  
            self.parent_node_type_embeddings = self.compute_parent_node_types_tensor(self.placeholders["node_types"], self.weights["node_type_embedding_lookup"])
            self.children_node_type_embeddings = self.compute_children_node_types_tensor(self.parent_node_type_embeddings, self.placeholders["children_indices"], self.node_type_dim)
            
            self.parent_node_embeddings = self.parent_node_type_embeddings
            self.children_embeddings = self.children_node_type_embeddings
            """Tree based Convolutional Layer"""
            # Example with batch size = 12 and num_conv = 8: shape = (12, 48, 128, 8)
            # Example with batch size = 1 and num_conv = 8: shape = (1, 48, 128, 8)
            self.conv_output = self.conv_layer(self.parent_node_embeddings, self.children_embeddings, self.placeholders["children_indices"], self.num_conv, self.node_dim)

            """The Primary Variable Capsule Layer."""
            # shape = (1, batch_size x max_tree_size, num_output, num_conv)
            # Example with batch size = 12: shape = (12, 48, 128, 8)
            # Example with batch size = 1: shape = (1, 48, 128, 8)
            self.primary_variable_caps = self.primary_variable_capsule_layer(self.conv_output)

            self.code_caps = self.dynamic_routing_code_caps(self.weights["w_dynamic_routing_code_caps"], self.weights["b_dynamic_routing_code_caps"], 
                                                self.primary_variable_caps, output_size=self.code_caps_num_caps, output_dimension=self.code_caps_output_dimension, 
                                                input_dimension=self.num_conv)
            # batch size = 1: (12, 80, 8, 1)
            self.code_caps = tf.squeeze(self.code_caps, axis=1)

            
            self.class_caps = self.dynamic_routing_class_caps(self.weights["w_dynamic_routing_class_caps"], 
                                    self.weights["b_dynamic_routing_class_caps"], self.code_caps, 
                                    output_size=self.class_caps_num_caps, output_dimension=self.class_caps_output_dimension, 
                                    input_dimension=self.code_caps_output_dimension)
            # batch size = 12: (12, num_class_cap, 50, 1)
            self.class_caps = tf.squeeze(self.class_caps, axis=1)
            
         
            # (12, 10, 1, 1)
            # # """Obtaining the classification output."""
            v_length = tf.sqrt(reduce_sum(tf.square(self.class_caps),axis=2, keepdims=True) + 1e-9)
            self.class_caps_prob = tf.reshape(v_length,(-1,self.label_size))
            self.softmax = tf.nn.softmax(self.class_caps_prob)
            self.loss = self.loss_layer(self.class_caps_prob, self.placeholders["labels"])


    def squash(self, vector):
        vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
        vec_squashed = scalar_factor * vector  # element-wise
        return(vec_squashed)


    def dynamic_routing_code_caps(self, w, b, primary_variable_caps, output_size=80, output_dimension=16, input_dimension=8):
            # input_dimension: dimension of input capsule
            # output_dimension: dimension of output capsule
            # output_size: number of capsules

            
            # (batch_size, output_size x max_node, 1, num_conv, 1)
            primary_variable_caps_reshaped = tf.reshape(primary_variable_caps, shape=[self.batch_size, -1, 1, input_dimension, 1])


            w_dynamic_routing = w 
            b_dynamic_routing = b
            
            # tile with max_node = (1, output_size x max_node, 640, 2, 1)
            # (1, output_size x max_node, output_dimension x output_size, num_conv, 1)
            w_dynamic_routing = tf.tile(w_dynamic_routing, [1, tf.shape(primary_variable_caps_reshaped)[1], 1, 1, 1])

            
            # (batch_size, output_size x max_node, output_dimension * output_size, num_conv, 1)
            primary_variable_caps_tiled = tf.tile(primary_variable_caps_reshaped, [1, 1, output_dimension * output_size, 1, 1])

            # (3, output_size x max_node, output_dimension x output_size, num_conv, 1)
            u_hat = w_dynamic_routing * primary_variable_caps_tiled
            # (3, output_size x max_node, output_dimension x output_size, 1, 1)
            u_hat = tf.reduce_sum(u_hat, axis=3, keep_dims=True)
            
            # (3, output_size x max_node, output_size, output_dimension, 1)
            u_hat = tf.reshape(u_hat, shape=[-1, tf.shape(primary_variable_caps_reshaped)[1], output_size, output_dimension, 1])
            u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
            
            # # (3, output_size x max_node, output_size, 1, 1)
            delta_IJ = tf.zeros([self.batch_size, tf.shape(primary_variable_caps_reshaped)[1], output_size, 1, 1], dtype=tf.dtypes.float32)
            for r_iter in range(iter_routing):
                with tf.variable_scope('iter_' + str(r_iter)):
                    gamma_IJ = tf.nn.softmax(delta_IJ, axis=2)

                    if r_iter == iter_routing - 1:
                        # (3, output_size x max_node, output_size, output_dimension, 1)
                        s_J = tf.multiply(gamma_IJ, u_hat)
                        # (3, 1, 2 x 80, 8, 1)
                        s_J_1 = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                        v_J = self.squash(s_J_1)
                    elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                        # (3, output_size x max_node, output_size, output_dimension, 1)
                        s_J = tf.multiply(gamma_IJ, u_hat_stopped)
                        # (3, 1, 2 x 80, 8, 1)
                        s_J_1 = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                        # (3, 1, 2 x 80, 8, 1)
                        v_J = self.squash(s_J_1)
                        # (3, output_size x max_node, output_size, output_dimension, 1)
                        v_J_tiled = tf.tile(v_J, [1,tf.shape(primary_variable_caps_reshaped)[1], 1, 1, 1])
                        # (3, output_size x max_node, output_size, 1, 1)
                        u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                        # (3, output_size x max_node, output_size, 1, 1)
                        delta_IJ += u_produce_v
            
            return v_J


    def dynamic_routing_class_caps(self, w, b, primary_variable_caps, output_size=80, output_dimension=16, input_dimension=8):
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
        primary_variable_caps_reshaped = tf.reshape(primary_variable_caps, shape=[self.batch_size, -1, 1, input_dimension, 1])
        
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

        delta_IJ = tf.zeros([self.batch_size, tf.shape(primary_variable_caps_reshaped)[1], output_size, 1, 1], dtype=tf.dtypes.float32)
        for r_iter in range(iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                gamma_IJ = tf.nn.softmax(delta_IJ, axis=2)

                if r_iter == iter_routing - 1:
                    # (3, output_size x max_node, output_size, output_dimension, 1)
                    s_J = tf.multiply(gamma_IJ, u_hat)
                    # (3, 1, 80, 8, 1)
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                    v_J = self.squash(s_J)
                elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                    # (3, output_size x max_node, output_size, output_dimension, 1)
                    s_J = tf.multiply(gamma_IJ, u_hat_stopped)
                    # (3, 1, 80, 8, 1)
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + b_dynamic_routing
                    # (3, 1, 80, 8, 1)
                    v_J = self.squash(s_J)
                    # (3, output_size x max_node, output_size, output_dimension, 1)
                    v_J_tiled = tf.tile(v_J, [1,tf.shape(primary_variable_caps_reshaped)[1], 1, 1, 1])
                    # (3, output_size x max_node, output_size, 1, 1)
                    u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                    # (3, output_size x max_node, output_size, 1, 1)
                    delta_IJ += u_produce_v

        return v_J

    def primary_variable_capsule_layer(self, conv_output):
        primary_variable_caps= tf.concat(conv_output, axis=-1)
        return primary_variable_caps

    def compute_parent_node_types_tensor(self, nodes, node_embeddings_lookup):
        node_embeddings =  tf.nn.embedding_lookup(node_embeddings_lookup,nodes)
        return node_embeddings

    def nn_attention_layer(self, inputs,batch_size,mask,emb_size,channel_num):
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
        with tf.variable_scope("attention_primary_caps") as scope:
            inputs_ = tf.reshape(inputs, shape=[batch_size * N, emb_size * channel_num])  # (?*N, C*d)
            atten = tf.layers.dense(inputs_, units=int(emb_size * channel_num / 16), activation=tf.nn.tanh)  # (?*N, C*d/16)
            atten = tf.layers.dense(atten, units=channel_num, activation=None)  # (?*N, C)
            atten = tf.reshape(atten, shape=[batch_size, N, channel_num, 1])  # (?, N, C, 1)
            atten = self.mask_softmax(atten, mask, dim=1)  # (batch, N, C, 1)
            
            # (batch, N, 8, 128) x (batch, N, 8, 1)
            input_scaled = tf.multiply(inputs, atten)  # (batch, N, C, 1)
            num_nodes = tf.expand_dims(tf.reduce_sum(mask, axis=1, keep_dims=True), axis=-1)
            input_scaled = input_scaled * num_nodes

        return input_scaled

    def mask_softmax(self, inputs,mask,dim):
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


    def conv_node(self, parent_node_embeddings, children_embeddings, children_indices, node_dim, layer):
        """Perform convolutions over every batch sample."""
        with tf.name_scope('conv_node'):
            w_t, w_l, w_r = self.weights["w_t_" + str(layer)], self.weights["w_l_" + str(layer)], self.weights["w_r_" + str(layer)]
            b_conv = self.weights["b_conv_" + str(layer)]
       
            return self.conv_step(parent_node_embeddings, children_embeddings, children_indices, node_dim, w_t, w_r, w_l, b_conv)

    def conv_layer(self, parent_node_embeddings, children_embeddings, children_indices, num_conv, node_dim):
        with tf.name_scope('conv_layer'):
            # nodes = [
            #     tf.expand_dims(self.conv_node(parent_node_embeddings, children_embeddings, children_indices, node_dim, layer),axis=-1)
            #     for layer in range(num_conv)
            # ] 
            nodes = []
            for layer in range(num_conv):
                new_parent_embeddings = self.conv_node(parent_node_embeddings, children_embeddings, children_indices, node_dim, layer)
                children_embeddings = self.compute_children_node_types_tensor(new_parent_embeddings, children_indices, node_dim)
                nodes.append(tf.expand_dims(new_parent_embeddings, axis=-1))
            return nodes 

    def conv_step(self, parent_node_embeddings, children_embeddings, children_indices, node_dim, w_t, w_r, w_l, b_conv):
        """Convolve a batch of nodes and children.
        Lots of high dimensional tensors in this function. Intuitively it makes
        more sense if we did this work with while loops, but computationally this
        is more efficient. Don't try to wrap your head around all the tensor dot
        products, just follow the trail of dimensions.
        """
        with tf.name_scope('conv_step'):
            # nodes is shape (batch_size x max_tree_size x node_dim)
            # children is shape (batch_size x max_tree_size x max_children)

            with tf.name_scope('trees'):
              
                # add a 4th dimension to the parent nodes tensor
                # nodes is shape (batch_size x max_tree_size x 1 x node_dim)
                parent_node_embeddings = tf.expand_dims(parent_node_embeddings, axis=2)
                # tree_tensor is shape
                # (batch_size x max_tree_size x max_children + 1 x node_dim)
                tree_tensor = tf.concat([parent_node_embeddings, children_embeddings], axis=2, name='trees')

            with tf.name_scope('coefficients'):
                # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
                c_t = self.eta_t(children_indices)
                c_r = self.eta_r(children_indices, c_t)
                c_l = self.eta_l(children_indices, c_t, c_r)

                # concatenate the position coefficients into a tensor
                # (batch_size x max_tree_size x max_children + 1 x 3)
                coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

            with tf.name_scope('weights'):
                # stack weight matrices on top to make a weight tensor
                # (3, node_dim, output_size)
                weights = tf.stack([w_t, w_r, w_l], axis=0)

            with tf.name_scope('combine'):
                batch_size = tf.shape(children_indices)[0]
                max_tree_size = tf.shape(children_indices)[1]
                max_children = tf.shape(children_indices)[2]

                # reshape for matrix multiplication
                x = batch_size * max_tree_size
                y = max_children + 1
                result = tf.reshape(tree_tensor, (x, y, node_dim))
                coef = tf.reshape(coef, (x, y, 3))
                result = tf.matmul(result, coef, transpose_a=True)
                result = tf.reshape(result, (batch_size, max_tree_size, 3, node_dim))

                # output is (batch_size, max_tree_size, output_size)
                result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

                # output is (batch_size, max_tree_size, output_size)
                return tf.nn.tanh(result + b_conv)

    def compute_children_node_types_tensor(self, parent_node_embeddings, children_indices, node_type_dim):
        """Build the children tensor from the input nodes and child lookup."""
    
        max_children = tf.shape(children_indices)[2]
        batch_size = tf.shape(parent_node_embeddings)[0]
        num_nodes = tf.shape(parent_node_embeddings)[1]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, node_type_dim))
        # vector_lookup is (batch_size x num_nodes x node_dim)
        vector_lookup = tf.concat([zero_vecs, parent_node_embeddings[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children_indices = tf.expand_dims(children_indices, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children_indices = tf.concat([batch_indices, children_indices], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x node_type_dim)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children_indices)

    def eta_t(self, children):
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

    def eta_r(self, children, t_coef):
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

    def eta_l(self, children, coef_t, coef_r):
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

 
    def loss_layer(self, logits_node, labels):
        """Create a loss layer for training."""

        with tf.name_scope('loss_layer'):
            max_l = tf.square(tf.maximum(0., 0.9 - logits_node))
            max_r = tf.square(tf.maximum(0., logits_node - 0.1))
            T_c = labels
            L_c = T_c * max_l + 0.5 * (1 - T_c) * max_r
            
            loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

            return loss

    # def out_layer(self, logits_node):
    #     """Apply softmax to the output layer."""
    #     with tf.name_scope('output'):
    #         return tf.nn.softmax(logits_node)
