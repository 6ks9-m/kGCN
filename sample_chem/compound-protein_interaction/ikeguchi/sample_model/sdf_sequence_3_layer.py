import tensorflow as tf
import numpy as np
import joblib
import layers
import tensorflow.keras.layers as klayer

import sample_chem.multimodal.sample_model.model_utility as mu
import tensorflow.contrib.keras as K
from tensorflow.python.keras import backend as backend_K
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops

class SDFSequenceNetwork(mu.MultiModalNetwork):
    def __init__(self,
                 *args,
                 **kwargs,
    ):
        super(SDFSequenceNetwork, self).__init__(*args, **kwargs)

    @classmethod
    def build_placeholders(cls, info, config, batch_size=4):
        adj_channel_num = info.adj_channel_num
        placeholders = {
            'adjs':[[tf.sparse_placeholder(tf.float32, name="adj_"+str(a)+"_"+str(b))
                     for a in range(adj_channel_num)] for b in range(batch_size)],
            'nodes': tf.placeholder(tf.int32, shape=(batch_size, info.graph_node_num), name="node"),
            'labels': tf.placeholder(tf.float32, shape=(batch_size, info.label_dim), name="label"),
            'mask': tf.placeholder(tf.float32, shape=(batch_size,), name="mask"),
            'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
            'sequences': tf.placeholder(tf.int32,shape=(batch_size, info.sequence_max_length), name="sequences"),
            'sequences_len': tf.placeholder(tf.int32, shape=(batch_size, 2), name="sequences_len"),
            'mask_label': tf.placeholder(tf.float32, shape=(batch_size, info.label_dim), name="mask_label"),
            'is_train': tf.placeholder(tf.bool, name="is_train"),
            'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
        }
        if info.feature_enabled:
            placeholders['features'] = tf.placeholder(tf.float32, shape=(batch_size, info.graph_node_num,
                                                                       info.feature_dim), name="feature")
        else:
            placeholders['features'] = None
        placeholders['embedded_layer'] = tf.placeholder(tf.float32, shape=(batch_size, info.sequence_max_length,
                                                                           info.sequence_symbol_num,), name="embedded_layer")
        cls.placeholders = placeholders
        return placeholders    
        
    def build_model(self):
        ## aliases
        batch_size = self.batch_size
        in_adjs = self.placeholders["adjs"]
        features = self.placeholders["features"]
        sequences = self.placeholders["sequences"]
        sequences_len = self.placeholders["sequences_len"]
        in_nodes = self.placeholders["nodes"]
        labels = self.placeholders["labels"]
        mask = self.placeholders["mask"]
        dropout_rate = self.placeholders["dropout_rate"]
        mask_label = self.placeholders["mask_label"]
        is_train = self.placeholders["is_train"] 
        enabled_node_nums = self.placeholders["enabled_node_nums"]
        embedded_layer = self.placeholders['embedded_layer']

        layer = features
        print("graph input layer:",layer.shape)
        layer = layers.GraphConv(100, self.adj_channel_num)(layer, adj=in_adjs)
        layer = layers.GraphBatchNormalization()(layer, max_node_num=self.graph_node_num,
                                                 enabled_node_nums=enabled_node_nums)
        layer = tf.nn.relu(layer)
        
        layer = layers.GraphConv(100, self.adj_channel_num)(layer, adj=in_adjs)
        layer = layers.GraphBatchNormalization()(layer, max_node_num=self.graph_node_num,
                                                 enabled_node_nums=enabled_node_nums)
        layer = tf.nn.relu(layer)

        layer = layers.GraphConv(100, self.adj_channel_num)(layer, adj=in_adjs)
        layer = layers.GraphBatchNormalization()(layer, max_node_num=self.graph_node_num,
                                                 enabled_node_nums=enabled_node_nums)
        layer = tf.nn.relu(layer)

        layer = layers.GraphDense(100)(layer)
        layer = tf.nn.relu(layer)
        layer = layers.GraphGather()(layer)
        graph_output_layer = layer
        print("graph output layer:",graph_output_layer.shape)
        graph_output_layer_dim = 100

        with tf.variable_scope("seq_nn") as scope_part:
            # Embedding
            if self.feed_embedded_layer:
                layer = embedded_layer
            else:
                layer = self._embedding(sequences)                
            print("sequence input layer:",layer.shape)
            # CNN + Pooling
            stride = 1
            layer = tf.keras.layers.Conv1D(100, stride, padding="same",
                                                         activation='relu')(layer)
            layer = tf.keras.layers.MaxPooling1D(stride)(layer)

            layer = tf.keras.layers.Conv1D(100, stride, padding="same",
                                                         activation='relu')(layer)
            layer = tf.keras.layers.MaxPooling1D(stride)(layer)
        
            layer = tf.keras.layers.Conv1D(100, stride, padding="same",
                                                         activation='relu')(layer)
            layer = tf.keras.layers.MaxPooling1D(stride)(layer)

            layer = tf.keras.layers.Conv1D(1, stride,padding="same",
                                                         activation='tanh')(layer)
            layer = tf.squeeze(layer)
            
            if len(layer.shape) == 1:
                # When batch-size is 1, this shape doesn't have batch-size dimmension due to just previous 'tf.squeeze()'.
                layer = tf.expand_dims(layer, axis=0)
            seq_output_layer = layer
            seq_output_layer_dim = layer.shape[1]
            print("sequence output layer:",seq_output_layer.shape)

        layer = tf.concat([seq_output_layer, graph_output_layer], axis=1)
        print("shared_part input:",layer.shape)

        input_dim = seq_output_layer_dim + graph_output_layer_dim

        with tf.variable_scope("shared_nn") as scope_part:
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.Dense(52)(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.nn.relu(layer)

        layer = tf.keras.layers.Dense(self.label_dim)(layer)
	
        prediction=tf.nn.softmax(layer)
        # computing cost and metrics
        cost=mask*tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=layer)
        cost_opt=tf.reduce_mean(cost)

        metrics={}
        cost_sum=tf.reduce_sum(cost)

        correct_count=mask*tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)),tf.float32)
        metrics["correct_count"]=tf.reduce_sum(correct_count)
        return self, prediction, cost_opt, cost_sum, metrics
    
    def _embedding(self, sequences):
        y = tf.keras.layers.Embedding(self.sequence_symbol_num,
                                      self.embedding_dim)(sequences)
        return y

    def embedding(self, data=None):
        key = self.placeholders['sequences']
        feed_dict = {key: data}
        sequences = self.placeholders["sequences"]
        embedd_values = self._embedding(sequences)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(embedd_values, feed_dict)
            return out
    
build_placeholders = SDFSequenceNetwork.build_placeholders

def build_model(placeholders, info, config, batch_size=4, feed_embedded_layer=False):
    net = SDFSequenceNetwork(batch_size,
                             info.input_dim,
                             info.adj_channel_num,
                             info.sequence_symbol_num,
                             info.graph_node_num,
                             info.label_dim,
                             info.pos_weight,
                             feed_embedded_layer=feed_embedded_layer)
    return net.build_model()
    
    
