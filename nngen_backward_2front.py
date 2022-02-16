#coding:utf-8
#预测多或预测少的影响一样
#0导入模块，生成数据集
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import math
import matplotlib as mpl
import nngen_forward_2front
mpl.rcParams['font.family'] = 'sans-serif'  #config the matplot
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
BATCH_SIZE = 1
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99  # 学习衰减率
REGULARIZER = 0.0001
STEPS = 200000  #2000000
a = 1
train_flag=0
MODEL_SAVE_PATH = "./model_2front/"
MODEL_NAME="2front_model"
##############################读取训练数据##########################################
number_of_each_group = 15  # 11为4个输入点，1个答案点，一个model
number_input=10
#############################训练数据##############################################
train_data_nn1 = np.loadtxt('train_data_output3.txt', delimiter=',', unpack=True)
#调用本级目录直接写，调用上级目录下其他目录里面的文件like this data = np.loadtxt("../data/numeric.txt", dtype=np.int32, delimiter=',', skiprows=0, encoding='utf-8')
# train_data_nn1 = np.loadtxt('/home/lupeng/Downloads/anntest1/train_data_output_2019071801.txt', delimiter=',', unpack=True)

train_data_nn_len=int(len(train_data_nn1))
train_data_nn_group=int(train_data_nn_len/number_of_each_group)  #一组15个数
train_data_nn=[]
y_ref=[]
test_data_nn=[]
y_ref_test=[]
# for i in range(train_data_nn_group): #
#     train_data_nn.append([])
#     y_ref.append([])
#     for j in range(number_input):
#         train_data_nn[i].append(train_data_nn1[i*number_of_each_group+j])
#     # for k in range(4):  #只加了两个点进来，模式没有加进来
#     #     y_ref[i].append(train_data_nn1[i*number_of_each_group+number_input+k])
#     lable=[0,0,0,0,0,0,0,0,0,0]
#     for m in range(10):#分类的个数
#         if train_data_nn1[i*number_of_each_group+14]==m+1:
#             lable[m]=1
#     for n in range(10):
#         y_ref[i].append(lable[n])
train_data_nn_test1=[]
y_ref_test1=[]

for i in range(10):
    num_flag=0
    m = 0
    train_data_nn_test1.append([])
    y_ref_test.append([])

    while(num_flag<931): #931为内部最模式最多的

        m=m % train_data_nn_group
        if train_data_nn1[m*number_of_each_group+14]==i+1:

            train_data_nn_test1[len(train_data_nn_test1) - 1].append([])
            for k in range(number_of_each_group): #一共15个数，7个点，加一个模式
                train_data_nn_test1[len(train_data_nn_test1) - 1][num_flag].append(train_data_nn1[m*number_of_each_group+k])
            num_flag=num_flag+1
        m=m+1


train_data=[]
for i in range(10): #10个分类
    for j in range(931):  #931为内部最模式最多的
        train_data.append([])
        train_data[len(train_data)-1]=train_data_nn_test1[i][j]
np.random.seed(116)  #按照随机种子打乱数据
np.random.shuffle(train_data)


len_train_data=len(train_data)
train_data_temp=[]
y_ref_temp=[]
for i in range(len_train_data):  #把数据分开，就是输入和标签分开
    train_data_temp.append([])
    y_ref_temp.append([])
    for j in  range(number_input):
        train_data_temp[len(train_data_temp)-1].append(train_data[i][j])
    y_ref_temp[len(y_ref_temp)-1].append(train_data[i][number_of_each_group-1])  #提取最后一个作为模式

len_train_data_all=len(train_data_temp)
len_train=int(0.85*len_train_data_all)
#训练集
train_data_nn_test1=train_data_temp[0:len_train]  #
y_ref_test2=y_ref_temp[0:len_train]
y_ref_test1=[]
for i in range(len(y_ref_test2)):
    y_ref_test1.append([])
    lable=[0,0,0,0,0,0,0,0,0,0]
    for m in range(10):#分类的个数
        if y_ref_test2[i][0]==m+1:
            lable[m]=1
    for n in range(10):#分类的个数
        y_ref_test1[i].append(lable[n])
    yyy=1



#测试集
test_data_nn_test1=train_data_temp[len_train:len_train_data_all]  #
y_test=y_ref_temp[len_train:len_train_data_all]


# train_data_nn_test1=[]
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[0] #1
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[99] #2
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[98]#3
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[105]#4
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[130] #5
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[109] #6
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[129]#7
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[108]#8
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[92]#9
# train_data_nn_test1.append([])
# train_data_nn_test1[len(train_data_nn_test1)-1]=train_data_nn[91]#10


#
#
# y_ref_test1=[]
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[0] #1
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[99] #2
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[98]#3
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[105]#4
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[130] #5
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[109] #6
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[129]#7
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[108]#8
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[92]#9
# y_ref_test1.append([])
# y_ref_test1[len(y_ref_test1)-1]=y_ref[91]#10
#
#
#
# for i in  range(len(y_ref)):
#     if y_ref[i][4]==222:
#         u=i
#         break

i=0
ii=0
# for jjj in range(train_data_nn_group-346,train_data_nn_group): #
#     test_data_nn.append([])
#     y_ref_test.append([])
#     for j in range(number_input):
#
#         test_data_nn[i].append(train_data_nn1[jjj*number_of_each_group+j])
#     i = i + 1
#     for k in range(5):#两个点坐标＋一个模式
#         y_ref_test[ii].append(train_data_nn1[jjj*number_of_each_group+number_input+k])
#     ii=ii+1
#
# for i in  range(len(y_ref_test)):
#     if y_ref_test[i][4]==222:
#         u=i
#         break

#
# test_data_nn.remove(test_data_nn[n])  #u这一行数据有问题，需要删去
# y_ref_test.remove(y_ref_test[u])


y_ref_len=len(y_ref_test1)
train_data_nn_len=len(train_data_nn_test1)
aaa_max=0


def backward():
    error = 0
    #占位
    with tf.name_scope('inputs'):
        x_data = tf.placeholder(tf.float32, [None, nngen_forward_2front.INPUT_NODE], name='x_input')
        y_target = tf.placeholder(tf.float32, [None, nngen_forward_2front.OUTPUT_NODE], name='y_input')
    final_output = nngen_forward_2front.forward(x_data, REGULARIZER)
    #2定义损失函数及反向传播方法。
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, #学习基值
        global_step,
        (train_data_nn_len-1) / BATCH_SIZE,  #衰减速度,100为训练样本数
        LEARNING_RATE_DECAY,
        staircase=True)


    #定义损失函数为MSE,反向传播方法为梯度下降。
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y_target - final_output))  #y_target是个list，而final_output是arry，对应每行相减，最后求总和求平均
        #tf.scalar_summary('loss',loss)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()
    #3生成会话，训练STEPS轮
    with tf.Session() as sess:
        # merged = tf.merge_all_summaries()
        # writer = tf.train.SummaryWriter("logs_2/", sess.graph)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs7/", sess.graph)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        if train_flag==0:
            STEPS=len(test_data_nn_test1)
        for i in range(STEPS):            #for is 0-steps-1 ,has been vertify the i is 0- step-1,
            start = (i*BATCH_SIZE) % y_ref_len
            end = (i*BATCH_SIZE) % y_ref_len + BATCH_SIZE

            pp=train_data_nn_test1[start:end]  #取的数据的第0-2行，但不包含第二行，其实就是第0和第1行
            m=i%100
            xm=train_data_nn_test1[start:end]
            ym=y_ref[start:end]
            if train_flag==1:
                _,loss_v,step,output=sess.run([train_step,loss, global_step,final_output ], feed_dict={x_data: train_data_nn_test1[start:end], y_target: y_ref_test1[start:end]})

                if i % 50 == 0:
                    result = sess.run(merged,feed_dict={x_data: train_data_nn_test1[start:end], y_target: y_ref_test1[start:end]})
                    writer.add_summary(result, i)
                if i % 500 == 0:

                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                    print("After %d steps, loss is: %f" % (step, loss_v))
            else:
                output = sess.run(final_output, feed_dict={x_data: test_data_nn_test1[start:end]})
                output2=output[0]
                aaa=output2.tolist()
                aaa_max=aaa.index(max(aaa))
                aaa_max=aaa_max+1 #实际的加1才是真实的
                print("*********************")
                print("input",test_data_nn_test1[start:end])
                print( "daan",y_test[start:end])
                print("act_output",output)
                print("max", aaa_max)
                daan=int(y_test[start:end][0][0])
                if daan!=aaa_max:
                    error=error+1
                    print("error")
                    for j in range(4):  # 每提取一次会出现三条边，一个答案点，通过此循环，每次生成一条边
                        temp_taindata_out_x = []
                        temp_taindata_out_y = []
                        temp_taindata_out_x.append(test_data_nn_test1[start:end][0][2 * j + 0])  # 每组有11个数，5个点的xy坐标加一个model
                        temp_taindata_out_x.append(test_data_nn_test1[start:end][0][2 * j + 2])  #
                        temp_taindata_out_y.append(test_data_nn_test1[start:end][0][2 * j + 1])  #
                        temp_taindata_out_y.append(test_data_nn_test1[start:end][0][2 * j + 3])  #
                        plt.plot(temp_taindata_out_x, temp_taindata_out_y, 'g-s', linewidth=2, color='b',
                                 markerfacecolor='b', marker='o')

                    # plt.axis("equal")
                    plt.xlim((-1.5, 3.5))
                    plt.xticks(np.linspace(-2.5, 2.5, 5, endpoint=True))
                    plt.ylim((-1.5, 3.5))
                    plt.yticks(np.linspace(-2.5, 2.5, 5, endpoint=True))
                    plt.show()


        zhunquelv=error/len(y_test)
        print("准确率",1-zhunquelv)



def main():

    backward()

if __name__ == '__main__':
    main()
# direct to the local dir and run this in terminal:
# $ tensorboard --logdir logs


#在本代码#2中尝试其他反向传播方法，看对收敛速度的影响，把体会写到笔记中
