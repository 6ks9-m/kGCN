"""
□入力
・化合物:dragon
・タンパク質：profeat
□ネットワーク
・化合物：DNN
タンパク質：DNN
"""
import tensorflow as tf
import numpy as np
import joblib
import layers

import sample_chem.multimodal.sample_model.model_utility as mu
import tensorflow.contrib.keras as K


class DragonProfeatNetwork(mu.MultiModalNetwork):
    def __init__(self,
                 *args,
                 **kwargs,
    ):
        super(DragonProfeatNetwork, self).__init__(*args, **kwargs)

    @classmethod
    def build_placeholders(cls, info, config, batch_size=4):
        profeat_dim=info.vector_modal_dim[info.vector_modal_name["profeat"]]
        dragon_dim=info.vector_modal_dim[info.vector_modal_name["dragon"]]
        placeholders = {
            'labels': tf.placeholder(tf.float32, shape=(batch_size,info.label_dim),name="label"),
            'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
            'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
            'profeat': tf.placeholder(tf.float32, shape=(batch_size,profeat_dim),name="profeat"),
            'dragon': tf.placeholder(tf.float32, shape=(batch_size,dragon_dim),name="dragon"),
            'mask_label': tf.placeholder(tf.float32, shape=(batch_size,info.label_dim),name="mask_label"),
            'is_train': tf.placeholder(tf.bool, name="is_train"),
            'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
        }
        if info.feature_enabled:
            placeholders['features']=tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.feature_dim),name="feature")
        else:
            placeholders['features']=None
        cls.placeholders = placeholders
        return placeholders
    
    def build_model(self, info, batch_size=4):
        profeat_dim=info.vector_modal_dim[info.vector_modal_name["profeat"]]
        dragon_dim=info.vector_modal_dim[info.vector_modal_name["dragon"]]
        labels=self.placeholders["labels"]
        mask=self.placeholders["mask"]
        dropout_rate=self.placeholders["dropout_rate"]
        profeat = self.placeholders["profeat"]
        dragon = self.placeholders["dragon"]
        mask_label=self.placeholders["mask_label"]
        is_train=self.placeholders["is_train"] 
        enabled_node_nums=self.placeholders["enabled_node_nums"]
        
        ###
        ### Mol part
        ###
        layer=dragon    
        layer=K.layers.Dense(100)(layer)
        layer=K.layers.BatchNormalization()(layer)
        layer=tf.nn.relu(layer)
    
        graph_output_layer=layer
        graph_output_layer_dim=100
    
    
        ###
        ### Sequence part
        ###
        layer=profeat    
        layer=K.layers.Dense(100)(layer)
        layer=K.layers.BatchNormalization()(layer)
        layer=tf.nn.relu(layer)
        
        seq_output_layer=layer
        seq_output_layer_dim=100
       
    
        # #region 不要
        # input_dim = graph_output_layer_dim
        # layer = graph_output_layer
        # #endregion
    
        ###
        ### Shared part
        ##
        # 32dim (Graph part)+ 32 dim (Sequence part)
        
        layer=tf.concat([seq_output_layer,graph_output_layer],axis=1)
        input_dim=seq_output_layer_dim+graph_output_layer_dim
        
        layer=K.layers.Dense(100)(layer)
        layer=K.layers.BatchNormalization()(layer)
        layer=tf.nn.relu(layer)
    
        layer=K.layers.Dense(info.label_dim)(layer)
        # # 最終出力を作成
        # # shape：[12×50×2]
        logits = mu.multitask_logits(layer, labels.shape[1])
        model = logits
        # # costの計算　各タスクのバッチ数平均　12
        task_losses =  mu.add_training_loss(logits = logits,label = labels,pos_weight = info.pos_weight,\
                                            batch_size= batch_size,n_tasks = labels.shape[1],mask = mask_label)
        total_loss = tf.reduce_sum(task_losses)#全タスクのlossを合計
    
        ### multi-task loss
        cost_opt=task_losses
        each_cost = task_losses
    
        # 2値の確率予測：12×50×2
        prediction = mu.add_softmax(logits)
        prediction = tf.transpose(prediction,[1,0,2])
        metrics={}
        cost_sum= total_loss 
        # cost_sum = cost_opt
        metrics["each_cost"] = task_losses
    
        metrics["each_correct_count"] = [None]*labels.shape[1]
        for i in range(labels.shape[1]):
            equal_cnt=mask_label[:,i]*tf.cast(tf.equal(tf.cast(tf.argmax(prediction[:,i,:],1),tf.int16), tf.cast(labels[:,i],tf.int16)),tf.float32)
    
            each_correct_count=tf.cast(tf.reduce_sum(equal_cnt,axis=0),tf.float32)
            metrics["each_correct_count"][i] = each_correct_count
    
        # correct_count=0#mask*tf.cast(tf.reduce_all(tf.equal(tf.cast(tf.argmax(prediction,1),tf.int16), tf.cast(labels,tf.int16)),axis=1),tf.float32)
        metrics["correct_count"]= sum([metrics["each_correct_count"][i] for i in range(labels.shape[1])])
        return model,prediction,cost_opt,cost_sum,metrics
    
build_placeholders = DragonProfeatNetwork.build_placeholders

def build_model(placeholders, info, config, batch_size=4, feed_embedded_layer=False):
    net = DragonProfeatNetwork(batch_size,
                               info.input_dim,
                               info.adj_channel_num,
                               info.sequence_symbol_num,
                               info.graph_node_num,
                               info.label_dim,
                               info.pos_weight,
                               feed_embedded_layer=feed_embedded_layer)
    return net.build_model(info)
