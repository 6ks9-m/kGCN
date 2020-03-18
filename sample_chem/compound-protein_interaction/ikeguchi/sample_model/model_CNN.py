import tensorflow as tf
import numpy as np
import joblib
import layers
# from keras.layers import *
from keras import layers as klayer

import sample_model.model_utility as mu

def build_placeholders(info, batch_size=4, adj_channel_num=1,
                       embedding_dim=10):
    placeholders = {
        'labels':
        tf.placeholder(
            tf.float32, shape=(batch_size, info.label_dim), name="label"),
        'mask':
        tf.placeholder(tf.float32, shape=(batch_size, ), name="mask"),
        'dropout_rate':
        tf.placeholder(tf.float32, name="dropout_rate"),
        'sequences':
        tf.placeholder(
            tf.int32,
            shape=(batch_size, info.sequence_max_length),
            name="sequences"),
        'sequences_len':
        tf.placeholder(tf.int32, shape=(batch_size, 2), name="sequences_len"),
        'mask_label':
        tf.placeholder(
            tf.float32, shape=(batch_size, info.label_dim), name="mask_label"),
        'is_train':
        tf.placeholder(tf.bool, name="is_train"),
    }
    return placeholders

def build_model(placeholders,info,batch_size=4,adj_channel_num=1, embedding_dim=10):
    sequences = placeholders["sequences"]
    sequences_len = placeholders["sequences_len"]
    labels = placeholders["labels"]
    mask = placeholders["mask"]
    dropout_rate = placeholders["dropout_rate"]
    mask_label = placeholders["mask_label"]
    wd_b = None
    wd_w = 0.1
    is_train = placeholders["is_train"]
    dropout_rate = 1 - dropout_rate
    ###
    ### Sequence part
    ###
    with tf.variable_scope("seq_nn") as scope_part:
        # Embedding
        embedding_dim=25
        layer=layers.embedding_layer("embedding",sequences,info.sequence_symbol_num,embedding_dim,init_params_flag=True,params=None)
        # CNN + Pooling
        stride = 4
        layer=klayer.convolutional.Conv1D(505,stride,padding="same", activation='relu')(layer)
        layer=klayer.pooling.MaxPooling1D(stride)(layer)

        stride = 3
        layer=klayer.convolutional.Conv1D(200,stride,padding="same",activation='relu')(layer)
        layer=klayer.pooling.MaxPooling1D(stride)(layer)
        
        stride = 2
        layer=klayer.convolutional.Conv1D(100,stride,padding="same",activation='relu')(layer)
        layer=klayer.pooling.MaxPooling1D(stride)(layer)

        layer=klayer.convolutional.Conv1D(1,stride,padding="same",activation='tanh')(layer)
        
        layer = tf.squeeze(layer)

        output_dim=info.label_dim

    logits = mu.multitask_logits(layer, labels.shape[1])
    model = logits
    # # costの計算　各タスクのバッチ数平均　12
    task_losses =  mu.add_training_loss(logits = logits,label = labels,pos_weight = info.pos_weight,\
             batch_size= batch_size,n_tasks = labels.shape[1],mask = mask_label)
    total_loss = tf.reduce_sum(task_losses)  #全タスクのlossを合計

    ### multi-task loss
    cost_opt = task_losses
    each_cost = task_losses

    # 2値の確率予測：12×50×2
    prediction = mu.add_softmax(logits)

    metrics = {}
    cost_sum = total_loss
    # cost_sum = cost_opt
    metrics["each_cost"] = task_losses

    metrics["each_correct_count"] = {}
    for i in range(labels.shape[1]):
        equal_cnt = mask_label[:, i] * tf.cast(
            tf.equal(
                tf.cast(tf.argmax(prediction[i], 1), tf.int16),
                tf.cast(labels[:, i], tf.int16)), tf.float32)

        each_correct_count = tf.cast(
            tf.reduce_sum(equal_cnt, axis=0), tf.float32)
        metrics["each_correct_count"][i] = each_correct_count

    # correct_count=0#mask*tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.argmax(prediction,1),tf.int16), tf.cast(labels,tf.int16)),axis=1),tf.float32)
    metrics["correct_count"] = sum(
        [metrics["each_correct_count"][i] for i in range(labels.shape[1])])
    return model, prediction, cost_opt, cost_sum, metrics
