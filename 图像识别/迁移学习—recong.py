# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:56:18 2017

@author: Administrator
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import gc
from PIL import Image


'''模型及样本路径设置'''

BOTTLENECK_TENSOR_SIZE = 2048                          # 瓶颈层节点个数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'           # 瓶颈层输出张量名称
JPEG_DATA_TENSOR_NAME  = 'DecodeJpeg/contents:0'       # 输入层张量名称

MODEL_DIR  = './inception_dec_2015'                    # 模型存放文件夹
MODEL_FILE = 'tensorflow_inception_graph.pb'           # 模型名

CACHE_DIR  = './bottleneck_dogs'                       # 瓶颈输出中转文件夹
INPUT_DATA = './dogs'                                  # 数据文件夹

TEST_DATA  = './test1'                                 #  测试数据

VALIDATION_PERCENTAGE = 10                             # 验证用数据百分比
TEST_PERCENTAGE       = 10                             # 测试用数据百分比

'''新添加神经网络部参数设置'''

LEARNING_RATE = 0.04
STEP          = 5000
BATCH         = 200

def creat_image_lists(validation_percentage,testing_percentage):
    '''
    将图片(无路径文件名)信息保存在字典中
    :param validation_percentage: 验证数据百分比 
    :param testing_percentage:    测试数据百分比
    :return:                      字典{标签:{文件夹:str,训练:[],验证:[],测试:[]},...}
    '''
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 由于os.walk()列表第一个是'./'，所以排除
    is_root_dir = True            #<-----
    # 遍历各个label文件夹
    for sub_dir in sub_dirs:
        if is_root_dir:           #<-----
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list  = []
        dir_name   = os.path.basename(sub_dir)
        # 遍历各个可能的文件尾缀
        for extension in extensions:
            # file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_glob = os.path.join(sub_dir, '*.' + extension)
            file_list.extend(glob.glob(file_glob))      # 匹配并收集路径&文件名
            # print(file_glob,'\n',glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()                   # 生成label，实际就是小写文件夹名

        # 初始化各个路径&文件收集list
        training_images   = []
        testing_images    = []
        validation_images = []

        # 去路径，只保留文件名
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机划分数据给验证和测试
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (validation_percentage + testing_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # 本标签字典项生成
        result[label_name] = {
            'dir'        : dir_name,
            'training'   : training_images,
            'testing'    : testing_images,
            'validation' : validation_images
        }
    return result

def get_random_cached_bottlenecks(sess,n_class,image_lists,batch,category,jpeg_data_tensor,bottleneck_tensor):
    '''
    函数随机获取一个batch的图片作为训练数据
    :param sess: 
    :param n_class: 
    :param image_lists: 
    :param how_many: 
    :param category:            training or validation
    :param jpeg_data_tensor: 
    :param bottleneck_tensor: 
    :return:                    瓶颈张量输出 & label
    '''
    bottlenecks   = []
    ground_truths = []
    for i in range(batch):
        label_index = random.randrange(n_class)              # 标签索引随机生成
        label_name  = list(image_lists.keys())[label_index]  # 标签名获取
        image_index = random.randrange(65536)                # 标签内图片索引随机种子
        # 瓶颈层张量
        bottleneck = get_or_create_bottleneck(               # 获取对应标签随机图片瓶颈张量
            sess,image_lists,label_name,image_index,category,
            jpeg_data_tensor,bottleneck_tensor)
        ground_truth = np.zeros(n_class,dtype=np.float32)
        ground_truth[label_index] = 1.0                      # 标准结果[0,0,1,0...]
        # 收集瓶颈张量和label
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def get_or_create_bottleneck(
        sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    '''
    寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
    :param sess: 
    :param image_lists:       全图像字典
    :param label_name:        当前标签
    :param index:             图片索引
    :param category:          training or validation
    :param jpeg_data_tensor: 
    :param bottleneck_tensor: 
    :return: 
    '''
    label_lists  = image_lists[label_name]          # 本标签字典获取 标签:{文件夹:str,训练:[],验证:[],测试:[]}
    sub_dir      = label_lists['dir']               # 获取标签值
    sub_dir_path = os.path.join(CACHE_DIR,sub_dir)  # 保存文件路径
    if not os.path.exists(sub_dir_path):os.mkdir(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        #image_data = gfile.FastGFile(image_path,'rb').read()
        image_data = open(image_path,'rb').read()
        # print(gfile.FastGFile(image_path,'rb').read()==open(image_path,'rb').read())
        # 生成向前传播后的瓶颈张量
        bottleneck_values = run_bottleneck_on_images(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
#        print("ranhuiqian:",type(bottleneck_values))
#        print(bottleneck_values)
        # list2string以便于写入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        # print(bottleneck_values)
        # print(bottleneck_string)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回的是list注意
#    print("ranhui:",type(bottleneck_values))
#    print(bottleneck_values)
#    input()
    return bottleneck_values

def run_bottleneck_on_images(sess,image_data,jpeg_data_tensor,bottleneck_tensor):
    '''
    使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
    :param sess:              会话句柄
    :param image_data:        图片文件句柄
    :param jpeg_data_tensor:  输入张量句柄
    :param bottleneck_tensor: 瓶颈张量句柄
    :return:                  瓶颈张量值
    '''
    # print('input:',len(image_data))
    bottleneck_values = sess.run(bottleneck_tensor,feed_dict={jpeg_data_tensor:image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    # print('bottle:',len(bottleneck_values))
    return bottleneck_values

def get_bottleneck_path(image_lists, label_name, index, category):
    '''
    获取一张图片的中转（featuremap）地址(添加txt)
    :param image_lists:   全图片字典
    :param label_name:    标签名
    :param index:         随机数索引
    :param category:      training or validation
    :return:              中转（featuremap）地址(添加txt)
    '''
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

def get_image_path(image_lists, image_dir, label_name, index, category):
    '''
    通过类别名称、所属数据集和图片编号获取一张图片的中转（featuremap）地址(无txt)
    :param image_lists: 全图片字典
    :param image_dir:   外层文件夹（内部是标签文件夹）
    :param label_name:  标签名
    :param index:       随机数索引
    :param category:    training or validation
    :return:            图片中间变量地址
    '''
    label_lists   = image_lists[label_name]
    category_list = label_lists[category]       # 获取目标category图片列表
    mod_index     = index % len(category_list)  # 随机获取一张图片的索引
    base_name     = category_list[mod_index]    # 通过索引获取图片名
    return os.path.join(image_dir,label_lists['dir'],base_name)

def get_test_bottlenecks(sess,image_lists,n_class,jpeg_data_tensor,bottleneck_tensor):
    '''
    获取全部的测试数据,计算输出
    :param sess: 
    :param image_lists: 
    :param n_class: 
    :param jpeg_data_tensor: 
    :param bottleneck_tensor: 
    :return:                   瓶颈输出 & label
    '''
    bottlenecks  = []
    ground_truths = []
#    label_name_list = list(image_lists.keys())
#    label_name_list = tuple(label_name_list)    
#    print(type(image_lists))
#    print(image_lists)
#    input()
#    label_name_list = tuple(label_name_list)
    for label_index,label_name in enumerate(image_lists):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]): # 索引, {文件名}
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index,
                category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_class, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


class fine_tune:
    
    def __init__(self,images_lists,n_class,jpeg_data_tensor,bottleneck_tensor):
        self.n_class = n_class
        self.images_lists = images_lists
        self.jpeg_data_tensor = jpeg_data_tensor
        self.bottleneck_tensor = bottleneck_tensor
        self.creat_fine_tune_net()
    
    def creat_fine_tune_net(self):
        # 输入层,由原模型输出层feed
        self.bottleneck_input   = tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
        self.ground_truth_input = tf.placeholder(tf.float32,[None,self.n_class]               ,name='GroundTruthInput')
        # 全连接层
        with tf.name_scope('final_train_ops'):
            Weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,self.n_class],stddev=0.001),name="fine_tune_weights")
            biases  = tf.Variable(tf.zeros([self.n_class]),name="fine_tune_biases")

        self.logits = tf.matmul(self.bottleneck_input,Weights) + biases
        self.final_tensor = tf.nn.softmax(self.logits)
        # 交叉熵损失函数
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.ground_truth_input))
        # 优化算法选择
        self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.cross_entropy)
    
        # 正确率
        with tf.name_scope('evaluation'):
            self.correct_prediction = tf.equal(tf.argmax(self.final_tensor,1),tf.argmax(self.ground_truth_input,1))
            self.evaluation_step    = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
    
        self.sess = tf.Session()
#        init = tf.global_variables_initializer()
#        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, 'dogs_model/l')
        
    def train_fine_tune_net(self):
        for i in range(STEP):
            # 随机batch获取瓶颈输出 & label
            train_bottlenecks,train_ground_truth = get_random_cached_bottlenecks(
                    self.sess,self.n_class,self.images_lists,BATCH,'training',self.jpeg_data_tensor,self.bottleneck_tensor)
            self.sess.run(self.train_step,feed_dict={self.bottleneck_input:train_bottlenecks,self.ground_truth_input:train_ground_truth})
            
            # 每迭代100次运行一次验证程序
            if i % 100 == 0:
                
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                        self.sess, self.n_class, self.images_lists, BATCH, 'validation', self.jpeg_data_tensor, self.bottleneck_tensor)

                validation_accuracy = self.sess.run( self.evaluation_step, feed_dict={
                        self.bottleneck_input: validation_bottlenecks, self.ground_truth_input: validation_ground_truth})
                
                self.saver.save(self.sess, 'dogs_model/l', write_meta_graph=False)
                
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                          (i, BATCH, validation_accuracy * 100))
               
        test_bottlenecks,test_ground_truth = get_test_bottlenecks(
                self.sess,self.images_lists,self.n_class,self.jpeg_data_tensor,self.bottleneck_tensor)
        
#        print('train_type:',type(test_bottlenecks))
#        print('train_cell_type:',type(test_bottlenecks[0]))
#        input()
        self.test_accuracy = self.sess.run( self.evaluation_step, feed_dict={
                self.bottleneck_input:test_bottlenecks,self.ground_truth_input:test_ground_truth})
        
        print('Final test accuracy = %.1f%%' % (self.test_accuracy * 100))
        self.sess.close()
        
        
class dogs_test:
    
    def __init__(self,n_class,jpeg_data_tensor,bottleneck_tensor,label_list):

#        self.images_lists = images_lists
        self.n_class = n_class
        self.jpeg_data_tensor = jpeg_data_tensor
        self.bottleneck_tensor = bottleneck_tensor
        self.label_list = label_list
        self.creat_dogs_test_net()
        
        
    def creat_dogs_test_net(self):
        # 输入层,由原模型输出层feed
        self.bottleneck_input   = tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
        self.ground_truth_input = tf.placeholder(tf.float32,[None,self.n_class]               ,name='GroundTruthInput')
        # 全连接层
        with tf.name_scope('final_train_ops'):
            Weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,self.n_class],stddev=0.001),name="fine_tune_weights")
            biases  = tf.Variable(tf.zeros([self.n_class]),name="fine_tune_biases")

        self.logits = tf.matmul(self.bottleneck_input,Weights) + biases
        self.final_tensor = tf.nn.softmax(self.logits)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
#        init = tf.global_variables_initializer()
#        self.sess.run(init)
        self.saver.restore(self.sess, 'dogs_model/l')
        
        
        
    def test(self):
        txtName = "dogs_test_resul.txt"
        f = open(txtName, "a+")  
        i = 0
        for image_file in os.listdir(r"./test1"):              #listdir的参数是文件夹的路径
            image_path = os.path.join("./test1/",image_file)
            image_data = open(image_path,'rb').read()
            
            bottleneck_values = self.sess.run(self.bottleneck_tensor,feed_dict={self.jpeg_data_tensor:image_data})
            
            test_label_list = self.sess.run(self.final_tensor, feed_dict = {self.bottleneck_input: bottleneck_values})
            
            
            ll = np.squeeze(test_label_list)
            
            sorted_inds = [i[0] for i in sorted(enumerate(-ll), key=lambda x: x[1])]

            biggest_label = self.label_list[sorted_inds[0]]

            
            (shotname,extension) = os.path.splitext(image_file)
            
            new_context = biggest_label + '\t' + shotname + '\n'
            f.write(new_context)
            i = i + 1
            if i%100 == 0 :
                print(i)
            
        f.close()
            
        


def main():
    # 生成文件字典
    images_lists = creat_image_lists(VALIDATION_PERCENTAGE,TEST_PERCENTAGE)
    # 记录label种类(字典项数)
    n_class = len(images_lists.keys())
    # label种类转为list
    label_list = list(images_lists.keys())
    print(label_list)

    # 加载模型
#    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:   # 阅读器上下文
    with open(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:            # 阅读器上下文
        graph_def = tf.GraphDef()                                         # 生成图
        graph_def.ParseFromString(f.read())                               # 图加载模型
    # 加载图上节点张量(按照句柄理解)
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(             # 从图上读取张量，同时导入默认图
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

    '''fine_tune'''
#    fine_tune_for_dogs = fine_tune(images_lists,n_class,jpeg_data_tensor,bottleneck_tensor)
#    fine_tune_for_dogs.train_fine_tune_net()

    dogs_test_test = dogs_test(n_class,jpeg_data_tensor,bottleneck_tensor,label_list)
    dogs_test_test.test()
    
#    graph_def.close()
    

if __name__ == '__main__':
    main()