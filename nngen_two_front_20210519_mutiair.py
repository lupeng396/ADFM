# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import math

import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(6, 6))
import pandas as pd
import numpy as np

import time
import nngen_backward_2front
import nngen_forward_2front
time_start=time.time()
print ('time start',time_start)
TEST_INTERVAL_SECS = 1
import os

BATCH_SIZE = 1
# plt.figure(figsize=(9,9))
with open(r'../datainput/20191203_mutiair2.cas', 'r') as data_file:
# with open(r'/home/lupeng/Downloads/anntest1/nn_gen/TEXT_20210228_SANJIAOXING.cas', 'r') as data_file:
    lines = data_file.readlines()
    # the fist data is invalid,so all copies  are  to distinguish them from the nomal data
################################################3寻找第二次含有含有"(13"，这是从line中搜索是一个字符字符所在的行#########
face_ln = []
Pbest_num=[]
nn = 0

for line1 in lines:   #把lines中的每一行提出来，放在line1中
    nn = nn + 1   #记录行数
    zi = "(13"    #记录行数需要查找的字符串
    elements = line1.split()  #把每一行打散
    if zi in elements:     #打散的字符串中查找是否有需要的字符串
        face_ln.append(nn)  #如果有把，其相应的行数记录下来
        print (face_ln)

###################################################
filelen = len(lines)
K = 1
M = 1
Dimension = lines[4 - K].split()[3][:-2]  # 去掉最后的分号和括号
Dimension = int(Dimension)

Number_of_Faces = int(lines[10 - K].split()[6][
                      :-2])  # get the string of first ten row 第六个空格后的数据 ,but cancel the last 2 character,at last string turn into int
Number_of_Nodes = int(lines[7 - K].split()[5][:-2])

Number_of_Boundary_Faces = lines[11 - K].split()[5]  # 取第八行第四个空格后的字符串，取出最后两位，然后强制转化为int
Number_of_Boundary_Faces = int(Number_of_Boundary_Faces[:-2])

Number_of_Interior_Faces = lines[12 - K].split()[5]  #
Number_of_Interior_Faces = int(Number_of_Interior_Faces[:-2])

Number_of_Cells = lines[15 - K].split()[6]  #
Number_of_Cells = int(Number_of_Cells[:-2])

Node_x = []  # 点的x坐标
Node_y = []
Node_x_new = []
Node_y_new = []
New_node_nearby=[]
New_face_nearby=[]

Face_Node_Number_new = []
Face_Node_Index_new = []
Left_Cell_Index_new = []
Right_Cell_Index_new = []
style_new_node=8  #新生成点相交性判断以后的模式，有可能直接生成，有可能也是和左右直接相连

p2p = []
delay = 1
Face_Node_Number = []
Face_Node_Index = []

Left_Cell_Index = []
Right_Cell_Index = []

Cell_Node_Number = []
Cell_Node_Index = []

temp_Cell = []
temp_node_face = []
count = 0
columns = 4  # lie
cross_flag_1 =0
Node_Face = []
Cun_nearbynode_nearbynode = []
Cun_nearbynode_nearbynode_1 = []
train_data_output = []
traindata_out = []

node_to_face=[]
node_num=0
Node_x=[]
Node_y=[]  #为了防止后面调用出错，将原始点数据信息清空
# Node_Face=[]
m1 = []
n1 = []
display_flag=1
print_flag=1
model_num1=1
model_num2=1
model_num3=1
model_num4=1
model_num5=1
model_num6=1
model_num7=1
model_num8=1
model_num9=1
model_num10=1

class point(): #定义类
    def __init__(self,x,y):
        self.x=x
        self.y=y

def cross(p1,p2,p3):#跨立实验
    x1=p2.x-p1.x
    y1=p2.y-p1.y
    x2=p3.x-p1.x
    y2=p3.y-p1.y
    return x1*y2-x2*y1

def IsIntersec(p1,p2,p3,p4): #判断两线段是否相交
    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1.x,p2.x)>=min(p3.x,p4.x)    #矩形1最右端大于矩形2最左端
    and max(p3.x,p4.x)>=min(p1.x,p2.x)   #矩形2最右端大于矩形最左端
    and max(p1.y,p2.y)>=min(p3.y,p4.y)   #矩形1最高端大于矩形最低端
    and max(p3.y,p4.y)>=min(p1.y,p2.y)): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0 and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            if (p1.x != p3.x and p1.y != p3.y) and(p1.x != p4.x and p1.y != p4.y) and (p2.x != p3.x and p2.y != p3.y) and (p2.x != p4.x and p2.y != p4.y):
                D = 1  #对于端点来说,所有点不相等才是算相交
            else:
                D = 0

        else:
            D=0
    else:
        D=0            #相交为1，不相交为0
    return D

def Node_zuobiao(number):
    zuobiao = []
    zuobiao.append(Node_x_new[2 * number])
    zuobiao.append(Node_y_new[2 * number + 1])
    return zuobiao


####################################################################################################################3

# 整体平移函数
def Translation(x, y, x0, y0):  # x,y表示要任何要平移的点，x0，y0表示将以何点为圆心作为参考点
    x_new = x - x0
    y_new = y - y0
    return [x_new, y_new]


def Anti_Translation(x, y, x0, y0):  # x,y表示要任何要anti平移的点，x0，y0表示将以何点为圆心作为参考点
    x_new = x + x0
    y_new = y + y0
    return [x_new, y_new]


# 缩放函数，归一化处理x_start,y_start,x_end,y_end,x,y分别表示坐标系x轴起点与终点坐标值，xy为任意需要变换的点
def Scaling(x_start, y_start, x_end, y_end, x, y):  # 注意xy是经过平移的点
    len = math.sqrt(((x_end - x_start) ** 2) + ((y_end - y_start) ** 2))
    if len==0:
        # display()  #显示边界
        node_num_display(Node_x_new, Node_y_new)
        if print_flag==1:
            print(node0,node1,node2,node3,node4)
            print(Node_x_new[node1-1],Node_y_new[node1-1])
    x_new = (1 / len) * x

    y_new = (1 / len) * y
    return [x_new, y_new]


def Anti_Scaling(x_start, y_start, x_end, y_end, x, y):  # 注意xy是经过反旋转的的点
    len = math.sqrt(((x_end - x_start) ** 2) + ((y_end - y_start) ** 2))
    x_new = (len) * x
    y_new = (len) * y
    return [x_new, y_new]


def Rotation(x_end, y_end, x_rot, y_rot):  # x_end,y_end为第一步平移后face另外一个点的值以便求与全局坐标系求夹角ceta，
    # xrot yrot为任意需要旋转的坐标
    # 相当于勾股定理，求得斜线的长度
    x = np.array([x_end, y_end])  # 向量1  x'
    y = np.array([1, 0])  # 全局坐标系x
    jj = x.dot(x)  # 向量点乘
    Lx = np.sqrt(x.dot(x))  # 向量点乘
    Ly = np.sqrt(y.dot(y))  # 向量点乘
    kk = x.dot(y)  # 向量点乘
    cos_angle = x.dot(y) / (Lx * Ly)
    sin_angle = y_end / (Lx * Ly)
    # 说明https://zhidao.baidu.com/question/1705964907383367940.html
    x_new = round(cos_angle * x_rot + sin_angle * y_rot, 2)
    y_new = round(-sin_angle * x_rot + cos_angle * y_rot, 2)
    return [x_new, y_new]


def Anti_Rotation(x_end, y_end, x_rot, y_rot):  # x_end,y_end为第一步平移后face另外一个点的值以便求与全局坐标系求夹角ceta，x_rot,y_rot是需要反旋转的量
    # xrot yrot为任意需要旋转的坐标
    # 相当于勾股定理，求得斜线的长度
    x = np.array([x_end, y_end])  # 向量1  x'
    y = np.array([1, 0])  # 全局坐标系x
    jj = x.dot(x)  # 向量点乘
    Lx = np.sqrt(x.dot(x))  # 向量点乘
    Ly = np.sqrt(y.dot(y))  # 向量点乘
    kk = x.dot(y)  # 向量点乘
    cos_angle = x.dot(y) / (Lx * Ly)
    sin_angle = y_end / (Lx * Ly)
    # 说明https://zhidao.baidu.com/question/1705964907383367940.html
    x_new = round(cos_angle * x_rot - sin_angle * y_rot, 2)
    y_new = round(sin_angle * x_rot + cos_angle * y_rot, 2)

    return [x_new, y_new]
def nearby_combine(node1x,node1y,R,node2x,node2y):#判断node2x,node2y是不是在以node1x,node1y为圆心,R为半径的圆以内，是返回1，不是返回0
    if (node2x-node1x)**2+(node2y-node1y)**2<R:
        return 1
    else:
        return 0


# 111111######################Node_x,Node_y##################################################################################3
for i in range(Number_of_Nodes):  # 读取文件第22行开始的25个点的数据
    data = lines[i + 21].split()[0]  # 读取文件第22行的第0个空格后数据
    Node_x.append(float(data))  # 将data字符串形式转化为浮点型,点从x【0】开始
    data1 = lines[i + 21].split()[1]  # 读取文件第22行的空格第一个数据
    Node_y.append(float(data1))
#################################################################################################################################
# 2222222###################Face_Node_Number  Face_Node_Index  Left_Cell_Index  Right_Cell_Index#################################
for i in range(Number_of_Boundary_Faces):  # 16 boundary Face
    Face_Node_Number.append(2)

    data2 = lines[i + face_ln[2]].split()[0]  # face_ln[2]表示（13字符串出现第三次所在的行
    Face_Node_Index.append(int(data2, 16))

    data3 = lines[i + face_ln[2]].split()[1]  # 读取
    Face_Node_Index.append(int(data3, 16))

    data4 = lines[i + face_ln[2]].split()[2]  # 读取
    Left_Cell_Index.append(int(data4, 16))

    data5 = lines[i + face_ln[2]].split()[3]  # 读取
    Right_Cell_Index.append(int(data5, 16))


for i in range(Number_of_Interior_Faces):  # 16 boundary Face
    Face_Node_Number.append(2)

    data2 = lines[i + face_ln[1]].split()[0]  # face_ln[2]表示（13字符串出现第三次所在的行
    Face_Node_Index.append(int(data2, 16))

    data3 = lines[i + face_ln[1]].split()[1]  # 读取
    Face_Node_Index.append(int(data3, 16))

    data4 = lines[i + face_ln[1]].split()[2]  # 读取
    Left_Cell_Index.append(int(data4, 16))

    data5 = lines[i + face_ln[1]].split()[3]  # 读取
    Right_Cell_Index.append(int(data5, 16))

################由已知数据提取边界和相关点,构成阵面推进的数据结构###########


for i in range(Number_of_Boundary_Faces):  #60 boundary Face

    Face_Node_Index_new.append(Face_Node_Index[2*i])
    Face_Node_Index_new.append(Face_Node_Index[2 * i+1])
    Left_Cell_Index_new.append(-1)
    Right_Cell_Index_new.append(0)

for i in range(Number_of_Boundary_Faces):  #60 个点的数量

    Node_x_new.append(Node_x[i])
    Node_y_new.append(Node_y[i])



################################显示边界###################################3
def dispaly1():
    ########################以下显示边界点信息####################################
    for i in range(0, Number_of_Boundary_Faces):  # 16 boundary Face 只显示年华画的内部边界
        jkl = Face_Node_Index[2 * i]  # 存的一条边起点序号
        hhj = Face_Node_Index[2 * i + 1]  # 存的一条边终点序号
        gx = [Node_x[Face_Node_Index[2 * i] - 1], Node_x[Face_Node_Index[2 * i + 1] - 1]]  # 存的理论点号与实际存的点号相差1
        gy = [Node_y[Face_Node_Index[2 * i] - 1], Node_y[Face_Node_Index[2 * i + 1] - 1]]  # 存的理论点号与实际存的点号相差1

        plt.plot(gx, gy, 'g-s', color='g', markerfacecolor='g', marker='.')
    #########################以下显示内部边信息##################################
    for i in range(Number_of_Interior_Faces):  # 16 boundary Face
        jkl = Face_Node_Index[2 * (i + Number_of_Boundary_Faces)]  # 存的一条边起点序号
        hhj = Face_Node_Index[2 * (i + Number_of_Boundary_Faces) + 1]  # 存的一条边终点序号
        gix = [Node_x[Face_Node_Index[2 * (i + Number_of_Boundary_Faces)] - 1],
               Node_x[Face_Node_Index[2 * (i + Number_of_Boundary_Faces) + 1] - 1]]
        giy = [Node_y[Face_Node_Index[2 * (i + Number_of_Boundary_Faces)] - 1],
               Node_y[Face_Node_Index[2 * (i + Number_of_Boundary_Faces) + 1] - 1]]

        plt.plot(gix, giy, 'g--', color='g', markerfacecolor='g', marker='.')
    #############################以下显示所有的点号##################################
    x = np.array(Node_x_new) #Node_x_new 相互改
    y = np.array(Node_y_new)
    i = 0
    for a, b in zip(x, y):
        i = i + 1
        plt.annotate('%s' % (i), xy=(a, b), color='k', xytext=(0, 0), textcoords='offset points')
    plt.axis("equal")
    plt.show()
    return


aaa = 0
# dispaly1()
ccc = 0

node1_new_node_x = []
node1_new_node_y = []
new_node_node2_x = []
new_node_node2_y = []


# plt.figure(figsize=(100, 100))  # 固定输入照片的大小，900*900
def display():

    # def display(node11, node22, new_node):
    # for i in range(face_num):  #
    for i in range(Number_of_Boundary_Faces):  #
        node1 = Face_Node_Index_new[2 * i]  #
        node2 = Face_Node_Index_new[2 * i + 1]  #
        temp_taindata_x = [Node_x_new[node1 - 1], Node_x_new[node2 - 1]]
        temp_taindata_y = [Node_y_new[node1 - 1], Node_y_new[node2 - 1]]
        plt.plot(temp_taindata_x, temp_taindata_y, 'g-s', linewidth=2, color='g', markerfacecolor='g', marker='.')
        plt.axis("equal")
    # plt.text(1.6, 1.4, r'$\mu=numk,\ \sigma=15$')  # 文本中注释
    # plt.text(1.5, 1.4, "numk=%.0f"% numk, fontdict=None, withdash=False)
    node_num_display(Node_x_new, Node_y_new)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def display3():

    plt.plot(Node_x_new, Node_y_new, 'g-s', linewidth=2, color='g', markerfacecolor='g', marker='')

    plt.xlim(-90, 90)
    plt.ylim(-60, 60)
    # plt.show()   #用于显示，为了加快仿真，采用上方的保存方式，免得出现后去关闭




def node_num_display(Node_x_new, Node_y_new):
    x = np.array(Node_x_new)
    y = np.array(Node_y_new)
    i = 0
    for a, b in zip(x, y):
        i = i + 1
        plt.annotate('%s' % (i), xy=(a, b), fontsize=16,color='b', xytext=(0, 0), textcoords='offset points') #https://zhuanlan.zhihu.com/p/267064157
    c=0



def len_node2node(node1, node2):
    node1_x = Node_x_new[node1 - 1]
    node1_y = Node_y_new[node1 - 1]
    node2_x = Node_x_new[node2 - 1]
    node2_y = Node_y_new[node2 - 1]
    node12_len = math.sqrt(((node2_x - node1_x) ** 2) + ((node2_y - node1_y) ** 2))
    node12_len=round(node12_len,3)
    return node12_len

def vetor_angle(node1,node2,node3): #node1 node2是阵面构成点，node3是测试点
    node12 = node3
    node11 = node2  # 每单个阵面的起点

    x_shiliang1 = (Node_x_new[node12 - 1]-Node_x_new[node11 - 1])
    y_shiliang1 = (Node_y_new[node12 - 1]-Node_y_new[node11 - 1])

    node22 = node1
    node21 = node2 # 每单个阵面的起点

    x_shiliang2 = (Node_x_new[node22 - 1] - Node_x_new[node21 - 1])
    y_shiliang2 = (Node_y_new[node22 - 1] - Node_y_new[node21 - 1])
    # https://zengtaiping.blog.csdn.net/article/details/103593197     引用该方法
    OA_dot_OB=x_shiliang1*x_shiliang2+y_shiliang1*y_shiliang2
    OA_cross_OB=x_shiliang1*y_shiliang2-x_shiliang2*y_shiliang1
    angle=np.arctan2(OA_cross_OB,OA_dot_OB)
    aaa=angle*180/np.pi
    if aaa<0:
        aaa=360+aaa
    return aaa  #返回实际的向量adao

def node1_nearby_adv(node,face_num_temp,ref_node,node_start,node_end):  # 在阵面中寻找第一个点的临点，以及阵面的编号
    rr=[]
    num=0
    ref_node_x=ref_node[0]
    ref_node_y=ref_node[1]
    for i in range(len(AFT_stack)):
        if node == AFT_stack[i][1]:
            rr.append([])
            node_ref_len = math.sqrt(((Node_x_new[AFT_stack[i][0] - 1] - ref_node_x) ** 2) + ((Node_y_new[AFT_stack[i][0] - 1] - ref_node_y) ** 2))
            rr[num].append(AFT_stack[i][0])
            rr[num].append(AFT_stack[i][5])
            rr[num].append(node_ref_len)  #改变两个点的临点的临点有多个的情况，在某个点直接连接了三个阵面及以上的情况（因为空心存在）
            num=num+1
    rr = sorted(rr, key=lambda fff: fff[2])  # 按照第3个元素重新排序，让最短的阵面在最上方
    if len(rr)==0:
        a=[]
        b=[]
    else:
        # a=rr[0][0]
        # b=rr[0][1]
        if len(rr)==1:
            a=rr[0][0]
            b=rr[0][1]
        else:  # 含有多个直接相邻边的情况，比如内部有孔洞
            for i in range(len(rr)):
                nodei = rr[i][0]
                rr[i][2] = vetor_angle(nodei,node_start, node_end)  #因为程序里面对应的形参是node3--node2.到node1--node2 的逆时针，这里用的时候需要变一下
            rr = sorted(rr, key=lambda fff: fff[2])  # 按照第3个元素重新排序，让最短的阵面在最上方
            a = rr[0][0]
            b = rr[0][1]
    return[a,b]


def node2_nearby_adv(node,face_num_temp,ref_node,node_start,node_end):  # 在阵面中寻找第2个点的临点，以及阵面的编号
    rr = []
    num = 0
    ref_node_x = ref_node[0]
    ref_node_y = ref_node[1]
    for i in range(len(AFT_stack)):
        if node == AFT_stack[i][0]:
            rr.append([])
            # rr.append(Right_Cell_Index_new[(AFT_stack[i][5])-1]) # 防止出现一个点出现左右两边都有空的计算域，也算是分开计算域，就是不要选择三角形自己的边
            # tt.append(Right_Cell_Index_new[AFT_stack[0][5]-1]) #
            node_ref_len = math.sqrt(((Node_x_new[AFT_stack[i][1] - 1] - ref_node_x) ** 2) + (
                    (Node_y_new[AFT_stack[i][1] - 1] - ref_node_y) ** 2))
            rr[num].append(AFT_stack[i][1])
            rr[num].append(AFT_stack[i][5])
            rr[num].append(node_ref_len)  # 改变两个点的临点的临点有多个的情况，在某个点直接连接了三个阵面及以上的情况（因为空心存在）
            num = num + 1
    rr = sorted(rr, key=lambda fff: fff[2])  # 按照第3个元素重新排序，让最短的阵面在最上方
    if len(rr) == 0:
        a = []
        b = []
    else:
        if len(rr) == 1:
            a = rr[0][0]
            b = rr[0][1]
        else:#含有多个直接相邻边的情况，比如内部有孔洞
            for i in range(len(rr)):

                nodei=rr[i][0]
                rr[i][2]=vetor_angle(node_start,node_end, nodei)
            rr = sorted(rr, key=lambda fff: fff[2])  # 按照第3个元素重新排序，让最短的阵面在最上方
            a = rr[0][0]
            b = rr[0][1]
    return [a, b]

def del_adv_elemnt(adv, elemnt):  # 删除阵面中，第5个元素（face编号），删除阵面堆栈中的非活跃边，通过编号收缩取去删除
    two_len = len(adv)
    flag=0
    for i in range(two_len):
        if adv[i][5] == elemnt:
            adv.remove(adv[i])
            flag = 1
            break
    if flag==0:
        print ("没有要删除的阵面")
    return adv


def nearby_combine(node1x, node1y, R, node2x, node2y):  # 判断node2x,node2y是不是在以node1x,node1y为圆心,R为半径的圆以内，是返回1，不是返回0
    if (node2x - node1x) ** 2 + (node2y - node1y) ** 2 < (R*R):
        return 1
    else:
        return 0


def mid_advance(node1, node2):
    node1_x = Node_x_new[node1 - 1]
    node1_y = Node_y_new[node1 - 1]
    node2_x = Node_x_new[node2 - 1]
    node2_y = Node_y_new[node2 - 1]

def Triangle_Quality_Judgment(x1, y1, x2, y2, x3, y3):
    a = float(math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2))
    b = float(math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2))
    c = float(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    if y1==y2 and y2==y3: #三个点在一条直线的情况
        Q=0.0001
    else:
        s = (a + b + c) / 2  # 半周长
        S = (s * (s - a) * (s - b) * (s - c))  # 面积，海伦公式
        if S==0:
            S=0.001
        if S < 0:
            S = 0.001
        S = S ** 0.5  # 面积，海伦公式
        r=S/s   #参考https://blog.csdn.net/fourierFeng/article/details/14000915
        R=a*b*c/(4*S)
        Q=r/R
    return Q


def node_aft_len(p,a,b):   #p表示候选点，a表示临近阵面第一点，b表示临近阵面第二点，下面的b.x,b.y属于向量表示
    abx=b.x-a.x
    aby=b.y-a.y
    apx=p.x-a.x
    apy = p.y - a.y
    len_ab=abx*abx + aby*aby  #线段ab的长度的平方
    len_pa=math.sqrt((apx ** 2) + (apy ** 2))
    len_pb = math.sqrt(((p.x-b.x) ** 2) + ((p.y - b.y) ** 2))
    t=abx*apx+aby*apy
    t=t/len_ab
    if t<=0:
        return len_pa
    if t>=1:
        return len_pb
    if t>0 and t<1:
        ap = np.array([apx, apy, 0])
        ab = np.array([abx, aby, 0])

        len_ac = np.linalg.norm(np.cross(ap, ab) / np.linalg.norm(ab))  #求垂线距离
        return len_ac
def step2(node_x,node_y):  #求实际步长的
    temp1=[]
    temp2=[]
    inside_boundary_list=[]
    step_train_data=[]
    len_aft2aft_list=[]
    num_boundary=len(AFT_stack_in)
    len_AFT_in=len(AFT_stack_in[0])
    out_boundary = AFT_stack_out[0][4]  # 外部边等长，找0-49都可以
    for i in range(num_boundary):   #构建每个阵面的步长分布，按指数函数来进行,第49条外部face后才是内部face
        inside_boundary=AFT_stack_in[i][4] #内部根据实际的来，因为内部不均匀
        mid_x = (Node_x_new[AFT_stack_in[i][0] - 1] + Node_x_new[AFT_stack_in[i][1] - 1]) / 2  # 内部各个face的中心点
        mid_y = (Node_y_new[AFT_stack_in[i][0] - 1] + Node_y_new[AFT_stack_in[i][1] - 1]) / 2
        len_aft2aft=math.sqrt(((node_x-mid_x) ** 2) + ((node_y-mid_y) ** 2))
        inside_boundary_list.append(inside_boundary)
        len_aft2aft_list.append(len_aft2aft)
        # 严格来将 这里因该是每个内face中心到圆的距离，如果圆足够大，则不影响，我们这里影响大，因为外圆不够大，每个face到外圆距离差别大
        c=AFT_stack_in[i][len_AFT_in-1] #最后一位放ｉ内部阵面到外阵面的最短距离
        x=len_aft2aft/c  #将阵面的长度进行归一化处理
        # m=math.pow(out_boundary/inside_boundary,1) #0.1表示0.1涨到外部边那么长
        # y=(out_boundary-inside_boundary)*(1/c)*len_aft2aft+inside_boundary  #这里有个假设，就是具体最近阵面的为1时，阵面的生长长度就为外边界的长度了
        t = (math.exp(-2 * x) - 1) / (math.exp(-2) - 1)
        y = t * (out_boundary - inside_boundary) + inside_boundary
        # y=(math.exp(-2*x )-1)/(math.exp( -2)-1)(out_boundary-inside_boundary)+inside_boundary

        temp1.append(y)
    temp2 = min(temp1)
    # print("temp2=",temp2,"c=",c)
    min_index=temp1.index(min(temp1))

    inside_boundary_min=inside_boundary_list[min_index]
    len_aft2aft_min=len_aft2aft_list[min_index]

    ck=(out_boundary-inside_boundary_min)*len_aft2aft_min+inside_boundary_min
    # print("ck=", ck)
    return temp2

def step(node_x,node_y):  #求实际步长的
    temp1=[]
    test=[]
    num_boundary=len(boundary_stank)
    for i in range(50,num_boundary,1):   #构建每个阵面的步长分布，按指数函数来进行,第49条外部face后才是内部face
        test.append([])
        out_boundary=boundary_stank[0][4] #外部边等长，找0-49都可以
        inside_boundary=boundary_stank[i][4] #内部根据实际的来，因为内部不均匀
        test[len(test)-1].append(boundary_stank[i][0])
        test[len(test) - 1].append(boundary_stank[i][1])
        mid_x = (Node_x_new[boundary_stank[i][0] - 1] + Node_x_new[boundary_stank[i][1] - 1]) / 2  # 内部各个face的中心点
        mid_y = (Node_y_new[boundary_stank[i][0] - 1] + Node_y_new[boundary_stank[i][1] - 1]) / 2
        len_aft2aft=math.sqrt(((node_x-mid_x) ** 2) + ((node_y-mid_y) ** 2))
        # 严格来将 这里因该是每个内face中心到圆的距离，如果圆足够大，则不影响，我们这里影响大，因为外圆不够大，每个face到外圆距离差别大
        m=math.pow(out_boundary/inside_boundary,1) #0.1表示0.1涨到外部边那么长

        # y=inside_boundary*math.pow(m,len_aft2aft)
        y=out_boundary*len_aft2aft/8+inside_boundary  #这是除以8是因为距离的最大长度不再是以前的1了，现在半径是8，可以理解为归一化
        test[len(test) - 1].append(y)
        temp1.append(y)
    temp2=min(temp1)
    test = sorted(test, key=lambda fff: fff[2])  # 按照第5个元素重新排序，让最短的阵面在最上方
    return temp2


def judge(a,b,c,d):
  if min(a[0],b[0])<=max(c[0],d[0]) and min(c[1],d[1])<=max(a[1],b[1]) and min(c[0],d[0])<=max(a[0],b[0]) and min(a[1],b[1])<=max(c[1],d[1]):
      if a!=c and a!=d and b!=c and b!=d:
          return 1
      else:
        return 0
  else:
    return 0


def tangential_normal(x, y ):  # 求某个向量的切向到法相，x,y是线段的终点减去起点,本质上就是shun时针旋转90°
    cos_angle = 0
    sin_angle = -1  #sin(-90)
    x_new = round(cos_angle * x - sin_angle * y, 10)
    y_new = round(sin_angle *x + cos_angle * y, 10)
    return [x_new, y_new]


def add_AFT_stack(node1,node2,face_num,cell_num): #smooth_flag是否定方向，在凹凸角度大的地方不需要ｓｍｏｏｔｈ，０表示不光滑处理
    global AFT_stac
    AFT_stack.append([])
    len_aft = len(AFT_stack)
    AFT_stack[len_aft - 1].append(node1)
    AFT_stack[len_aft - 1].append(node2)
    AFT_stack[len_aft - 1].append(-1)  # -1
    AFT_stack[len_aft - 1].append(cell_num)  # 0.边界face的右边单元为0
    AFT_stack[len_aft - 1].append(len_node2node(node2, node1))  # 边界face的右边单元为0
    AFT_stack[len_aft - 1].append(face_num)  # 把编号存进去，因为后期会被打乱，所以存入编号


def add_face(node1,node2,left_cell,right_cell):
    global Face_Node_Index_new,Left_Cell_Index_new,Right_Cell_Index_new,face_num
    face_num = face_num + 1
    Face_Node_Index_new.append(node1)  # 新的阵面层的第i个点
    Face_Node_Index_new.append(node2)  # 旧的阵面层的第i个点  他们连起来就是一直线
    Left_Cell_Index_new.append(left_cell)
    Right_Cell_Index_new.append(right_cell)

def node_nearby(new_node,two_ponit_len,node_start,node_end):
    # 临近点的坐标存在New_node_nearby这个list里面
    global New_node_nearby,New_face_nearby,numk
    New_node_nearby=[]
    New_face_nearby=[]
    Counter_New_node_nearby=0
    for i in range(len(AFT_stack)):
        # def nearby_combine(node1x, node1y, R, node2x, node2y):  # 判断node2x,node2y是不是在以node1x,node1y为圆心,R为半径的圆以内，是返回1，不是返回0
        if nearby_combine(new_node[0], new_node[1], 1.5 * two_ponit_len, Node_x_new[AFT_stack[i][0] - 1],
                          Node_y_new[AFT_stack[i][0] - 1]):  # 返回值为1，说明了阵面有点在新生成点的附近

            New_node_nearby.append([])  # 建立了一个二维的数组，存放临近点，作所以做成2维，主要是因为后期的每个临近点要根据阵面质量来排序
            New_node_nearby[Counter_New_node_nearby].append(AFT_stack[i][0])
            Counter_New_node_nearby = Counter_New_node_nearby + 1
    for i in range(len(AFT_stack)):
        # def nearby_combine(node1x, node1y, R, node2x, node2y):  # 判断node2x,node2y是不是在以node1x,node1y为圆心,R为半径的圆以内，是返回1，不是返回0

        if nearby_combine(new_node[0], new_node[1], 1.5 * two_ponit_len, Node_x_new[AFT_stack[i][1] - 1],
                          Node_y_new[AFT_stack[i][1] - 1]):
            New_node_nearby.append([])
            New_node_nearby[Counter_New_node_nearby].append(AFT_stack[i][1])
            Counter_New_node_nearby = Counter_New_node_nearby + 1
    # 将New_node_nearby里面重复的点去掉放在list2，然后list2再赋值给New_node_nearby，然后还要去掉阵面上的两个点
    list2 = []
    for i in New_node_nearby:
        if i not in list2:
            nodei=i[0]
            nodei_xy_1 = Translation(Node_x_new[node_end - 1], Node_y_new[node_end - 1], Node_x_new[node_start - 1],
                                     Node_y_new[node_start - 1])
            nodei_xy = Translation(Node_x_new[nodei - 1], Node_y_new[nodei - 1], Node_x_new[node_start - 1],
                                   Node_y_new[node_start - 1])
            nodei_xy = Scaling(Node_x_new[node_start - 1], Node_y_new[node_start - 1], Node_x_new[node_end - 1],
                               Node_y_new[node_end - 1],
                               nodei_xy[0], nodei_xy[1])
            nodei_xy = Rotation(nodei_xy_1[0], nodei_xy_1[1], nodei_xy[0], nodei_xy[1])

            if (i != [node_start]) and (i != [node_end]) and nodei_xy[1]>0:  # 不能是阵面上的点
                list2.append(i)
    New_node_nearby = list2
    # 将新点坐标也放进去，后面会根据三角形生成质量整体排序
    New_node_nearby.append([])
    New_node_nearby[len(New_node_nearby) - 1].append(len(Node_x_new))  # len(Node_x_new)为最后一个点的号数，从1开始的,在找点的时候要对前值减一
    # 找新点附近的阵面，后面判断相交性；将新点的临近阵元的编号号存在New_face_nearby这个list里面，按新点与临近阵元中心
    for i in range(1, len(AFT_stack)):  # i从1开始，去掉了生成新点的阵面
        AFT_i_Centor_x = 0.5 * (Node_x_new[AFT_stack[i][0] - 1] + Node_x_new[AFT_stack[i][1] - 1])  # 阵面i的中心坐标x
        AFT_i_Centor_y = 0.5 * (Node_y_new[AFT_stack[i][0] - 1] + Node_y_new[AFT_stack[i][1] - 1])  # 阵面i的中心坐标y
        if nearby_combine(new_node[0], new_node[1], 2 * two_ponit_len,
                          AFT_i_Centor_x, AFT_i_Centor_y):  # 返回值为1，说明了阵面有点在新生成点的附近
            New_face_nearby.append(i)  # 存的在AFT中的编号
    # 网格质量排序，将二维list中所有的临近点与阵面AB两点构成的三角形生成的质量通过函数判断正三角形的质量系数排序
    # Triangle_Quality_Judgment(x1, y1, x2, y2, x3, y3): 内部参数分别是是三个点的xy坐标值

    for i in range(len(New_node_nearby)):
        test6 = New_node_nearby[i][0]
        test7 = node1
        test8 = node2
        test_temp = Triangle_Quality_Judgment(Node_x_new[New_node_nearby[i][0] - 1],
                                              Node_y_new[New_node_nearby[i][0] - 1],
                                              Node_x_new[node_start - 1], Node_y_new[node_start - 1], Node_x_new[node_end - 1],
                                              Node_y_new[node_end - 1])
        if New_node_nearby[i][0] == len(Node_x_new):  # 新生成最好的点需要乘以一个质量系数
            test_temp = 0.68* test_temp
        if New_node_nearby[i][0] <= 59:  # 新生成最好的点需要乘以一个质量系数
            test_temp = 0.8* test_temp  #边界上的点也要乘以质量系数
        New_node_nearby[i].append(test_temp)
    New_node_nearby = sorted(New_node_nearby, key=lambda fff: fff[1], reverse=True)  # 按第二个元素降序排序


def cross_judge(node_a,node_b):  #返回相交性判断以及选的最优点
    global New_node_nearby,New_face_nearby
    # sorted（key,reverse=True or Flase）在key后面可以指定一个参数，就是正序和降序的选择
    cross_flag = 0  # 默认不相交，如果相交会让他置1
    near_flag_1 = 0  # 默认不临近，如果候选点与阵面很接近，会让他置1
    on_aft_flag_1 = 0  # 默认不临近，如果候选点与阵面很接近，会让他置1
    Pbest_num=0

    # 以下是相交性判断
    node2_xy_11 = Translation(Node_x_new[node_b - 1], Node_y_new[node_b - 1], Node_x_new[node_a - 1], Node_y_new[node_a - 1])

    for i in range(len(New_node_nearby)):
        cross_flag = 0  # 默认不相交，如果相交会让他置1
        near_flag_1 = 0  # 默认不临近，如果候选点与阵面很接近，会让他置1
        on_aft_flag_1 = 0  # 默认不临近，如果候选点与阵面很接近，会让他置1

        nodeN_xy = Translation(Node_x_new[New_node_nearby[i][0] - 1], Node_y_new[New_node_nearby[i][0] - 1],
                               Node_x_new[node_a - 1],
                               Node_y_new[node_a - 1])
        nodeN_xy = Scaling(Node_x_new[node_a - 1], Node_y_new[node_a - 1], Node_x_new[node_b - 1],
                           Node_y_new[node_b - 1],
                           nodeN_xy[0], nodeN_xy[1])
        nodeN_xy = Rotation(node2_xy_11[0], node2_xy_11[1], nodeN_xy[0], nodeN_xy[1])
        if nodeN_xy[1] < 0:  # 新点要位于阵面上方,上面的归一化这一套就是为了让选择的新点位于阵面上方，对于下方的直接算相交，不选，选后面的
            cross_flag = 1
            continue

        for j in range(len(New_face_nearby)):

            test1 = AFT_stack[0][0]
            test2 = AFT_stack[0][1]
            test3 = New_node_nearby[i][0]
            test4 = AFT_stack[New_face_nearby[j]][0]
            test5 = AFT_stack[New_face_nearby[j]][1]
            test9 = AFT_stack[New_face_nearby[j]][0]
            test10 = AFT_stack[New_face_nearby[j]][1]

            p1 = point(Node_x_new[node_a - 1], Node_y_new[node_a - 1])  # node1
            p11 = point(Node_x_new[node_b - 1], Node_y_new[node_b - 1])  # node2
            p2 = point(Node_x_new[New_node_nearby[i][0] - 1],
                       Node_y_new[New_node_nearby[i][0] - 1])  # 相邻点质量排序点,new_node

            p3 = point(Node_x_new[AFT_stack[New_face_nearby[j]][0] - 1],
                       Node_y_new[AFT_stack[New_face_nearby[j]][0] - 1])  # 相邻边的ｎｏｄｅ１
            p4 = point(Node_x_new[AFT_stack[New_face_nearby[j]][1] - 1],
                       Node_y_new[AFT_stack[New_face_nearby[j]][1] - 1])  # 相邻边的ｎｏｄｅ２

            r_left = IsIntersec(p1, p2, p3, p4)
            r_right = IsIntersec(p11, p2, p3, p4)
            # if test3 != test4 and test3 != test5:  # 新点不是阵面上的点，这种都不等的情况就是说新点不是阵面上的点，要测算距离，而阵面上的点距离为零不用测
            #     len_node2aft = node_aft_len(p2, p3, p4)
            #     tempm = real_step / 3.5
            #     if len_node2aft < tempm:
            #         near_flag_1 = 1

            # if r_left == 1 or r_right == 1 or near_flag_1 == 1:
            if r_left == 1 or r_right == 1:
                cross_flag = 1  # 只要预选的某个点和一个临近阵面相交，则选下一个候选点从新循环
                break
            else:
                cross_flag = 0  # 质量最好的点和所有阵面都不想交，所以要跳出两个循环
                Pbest_num = New_node_nearby[i]

        if cross_flag == 0:
            Pbest_num = New_node_nearby[i]
            break

    if cross_flag == 1:  # 只要预选的某个点和一个临近阵面相交，则选下一个候选点从新循环
        if print_flag== 1:
            print("所有点都要和相邻阵面相交，可以换一个最短阵面来试，有些地方可以考虑挖一个洞")
            print("阵面左边点，阵面右边点"), AFT_stack[0][0], AFT_stack[0][1]

            print("新点xy", Node_x_new[AFT_stack[0][0]-1], Node_y_new[AFT_stack[0][0]-1])

        display()  # 显示边界
        node_num_display(Node_x_new, Node_y_new)
    return [cross_flag,Pbest_num]

def new_node(node1,node2,x,y):#这里的x y是不考虑的step的时候的新点的位置,这个函数时阵面出去的新点考虑了步长的时候的新点
    mid_aft_x = (Node_x_new[node1 - 1] + Node_x_new[node2 - 1]) / 2  # 阵面中心x，y坐标
    mid_aft_y = (Node_y_new[node1 - 1] + Node_y_new[node2 - 1]) / 2

    real_step = step(mid_aft_x, mid_aft_y)
    if real_step >= 0.196:
        real_step = 0.196
    Qqq = real_step * (2 / (3 ** 0.5)) / node12_len
    # print "Q=",Qqq
    new_node = [x, y * Qqq]  # 注意这里/除2是浮点型除法，整数除法是// [0.5, (3 ** 0.5 / 2) * Qqq]
    return new_node

def tangential_mid_point(x1,y1,x2,y2,real_step):
    x_shiliang = (x1 - x2)
    y_shiliang = (y1 - y2)

    Tangential = [x_shiliang, y_shiliang]
    Tangential_temp = tangential_normal(Tangential[0], Tangential[1])

    mid_x=(x1+x2)*0.5
    mid_y= (y1 + y2) * 0.5


    lan1 = math.sqrt((Tangential_temp[0] ** 2) + (Tangential_temp[1] ** 2))
    unit_np = np.array(Tangential_temp) / lan1  # 单位法向向量
    temp = unit_np * real_step
    tqq = np.array([mid_x, mid_y])
    bb = tqq + temp
    ref_x=bb[0]
    ref_y=bb[1]

    return [ref_x,ref_y]
def display_face(node1,node2):
    if display_flag==1:
        node0_node2_x = [Node_x_new[node1 - 1], Node_x_new[node2 - 1]]
        node0_node2_y = [Node_y_new[node1 - 1], Node_y_new[node2 - 1]]
        plt.plot(node0_node2_x, node0_node2_y, 'g-s', linewidth=2, color='g', markerfacecolor='g', marker='')
##############################(Advancing Front Technique#################################################


def face_len(x1,y1,x2,y2):
    facelen= math.sqrt(((x1-x2) ** 2) + ((y1 - y2) ** 2))
    return facelen

# def unit_direction_add(node1,node2):  #0928  node1 is the vetor start point the node2 is the end point
#     x1=node1[0]
#     y1 = node1[1]
#     x2=node2[0]
#     y2 = node2[1]
#     lan1 = math.sqrt(((x2-x1) ** 2) + ((y2-y1) ** 2))
#     unit_x=(x2-x1)/lan1
#     unit_y = (y2-y1) / lan1
#     node_new_x=x1+unit_x
#     node_new_y=y1+unit_y   #to get the normorlize position of node2
#     return [node_new_x,node_new_y]

def unit_direction_add(node1,node2):  #0928  node1 is the vetor start point the node2 is the end point
    x1=node1[0]
    y1 = node1[1]
    x2=node2[0]
    y2 = node2[1]
    lan1 = math.sqrt(((x2-x1) ** 2) + ((y2-y1) ** 2))
    unit_x=(x2-x1)/lan1
    unit_y = (y2-y1) / lan1
    node_new_x=x1+unit_x
    node_new_y=y1+unit_y   #to get the normorlize position of node2
    return [node_new_x,node_new_y]

def unit_direction(node1,node2):  #0928  node1 is the vetor start point the node2 is the end point
    x1=node1[0]
    y1 = node1[1]
    x2=node2[0]
    y2 = node2[1]
    lan1 = math.sqrt(((x2-x1) ** 2) + ((y2-y1) ** 2))
    unit_x=(x2-x1)/lan1
    unit_y = (y2-y1) / lan1
    return [unit_x,unit_y]

def is_in_oneaft(node1, node2): #返回值第一位是是否在一个阵面上，2位是返回的编号
    aft_len=len(AFT_stack)
    in_oneaft_flag=0
    face_num_back=0
    for i in range(aft_len):
        if AFT_stack[i][0]==node1 and AFT_stack[i][1]==node2:
            face_num_back=AFT_stack[i][5]
            in_oneaft_flag = 1
            break
    return [in_oneaft_flag,face_num_back]





AFT_stack = []  # 用于存放活动阵面的堆栈，构建二维数组，关系到数组的按元素大小存放，删除首个元素，添加元素，比较等操作格式：[起点，终点，左单元，右单元，阵面长度，编号]，因为所有活动阵面都没有左单元，因此设置为-1
#
# Out_Number_of_Boundary_Faces=86
# Inside_Number_of_Boundary_Faces=14
for i in range(Number_of_Boundary_Faces):
    AFT_stack.append([])
    AFT_stack[i].append(Face_Node_Index_new[2 * i+ 1]) #和以前不一样，反着存的，不然就像方形外面长了
    AFT_stack[i].append(Face_Node_Index_new[2 * i])
    AFT_stack[i].append(-1)  # -1
    AFT_stack[i].append(0)  # 0.边界face的右边单元为0
    AFT_stack[i].append(len_node2node(AFT_stack[i][0], AFT_stack[i][1]))  # 边界face的右边单元为0
    AFT_stack[i].append(i + 1)  # 把编号存进去，因为后期会被打乱，所以存入编号


boundary_stank=AFT_stack  #因为后面AFT_stack会打乱顺序以及删除，保留初始边界信息



# for i in range(Out_Number_of_Boundary_Faces,Number_of_Boundary_Faces):
#     AFT_stack.append([])
#     AFT_stack[i].append(Face_Node_Index_new[2 * i+ 1]) #和以前不一样，反着存的，不然就像方形外面长了
#     AFT_stack[i].append(Face_Node_Index_new[2 * i ])
#     AFT_stack[i].append(-1)  # -1
#     AFT_stack[i].append(0)  # 0.边界face的右边单元为0
#     AFT_stack[i].append(len_node2node(AFT_stack[i][0], AFT_stack[i][1]))  # 边界face的右边单元为0
#     AFT_stack[i].append(i + 1)  # 把编号存进去，因为后期会被打乱，所以存入编号


# AFT_stack= sorted(AFT_stack, key=lambda fff: fff[4])  #按照第5个元素重新排序，让最短的阵面在最上方






# JJ = len(AFT_stack)
#
#
# for i in range(410,JJ,1):   #把阵面1 2 的数据交换了，因为从pointwise导入的时候，没有按右手顺序存
#     temp=AFT_stack[i][0]
#     AFT_stack[i][0]=AFT_stack[i][1]
#     AFT_stack[i][1]=temp


AFT_stack_in=[]
AFT_stack_out=[]
for i in range(len(AFT_stack)):
    if i<=58:
        AFT_stack_out.append([])
        AFT_stack_out[len(AFT_stack_out)-1]=AFT_stack[i]
    else:
        AFT_stack_in.append([])
        AFT_stack_in[len(AFT_stack_in)-1]=AFT_stack[i]

len_aftin_2_aftout_list=[]
for i in range(len(AFT_stack_in)):
    mid_in_x = (Node_x_new[AFT_stack_in[i][0] - 1] + Node_x_new[AFT_stack_in[i][1] - 1]) / 2  # 内部各个face的中心点
    mid_in_y = (Node_y_new[AFT_stack_in[i][0] - 1] + Node_y_new[AFT_stack_in[i][1] - 1]) / 2
    for k in range(len(AFT_stack_out)):
        mid_out_x = (Node_x_new[AFT_stack_out[k][0] - 1] + Node_x_new[AFT_stack_out[k][1] - 1]) / 2  # 内部各个face的中心点
        mid_out_y = (Node_y_new[AFT_stack_out[k][0] - 1] + Node_y_new[AFT_stack_out[k][1] - 1]) / 2
        len_aftin_2_aftout = math.sqrt(((mid_out_x - mid_in_x) ** 2) + ((mid_out_y - mid_in_y) ** 2))
        len_aftin_2_aftout_list.append(len_aftin_2_aftout)
    temp3=min(len_aftin_2_aftout_list)
    AFT_stack_in[i].append(temp3)



k = 0
numk = 0
cell_num = 0
face_num = len(AFT_stack)
# dispaly1()  #显示原始ponitwise的图形
display()  # 显示边界
# node_num_display(Node_x_new, Node_y_new)
erro_test=6000#完成是1053
# AFT_stack = sorted(AFT_stack, key=lambda fff: fff[4])  # 按照第5个元素重新排序，让最短的阵面在最上方
numk_flag=3300
test_node=[]
a=0
time_s=time.time()
with tf.Graph().as_default() as g:
    x_data = tf.placeholder(tf.float32, [None, nngen_forward_2front.INPUT_NODE], name='x_input')

    final_output = nngen_forward_2front.forward(x_data, None)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(nngen_backward_2front.MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            while len(AFT_stack) > 0:

                nearby_combine_flag = 3
                cross_flag_1=3
                cross_flag_2 = 3
                style_new_node = 8  # 新生成点相交性判断以后的模式，有可能直接生成，有可能也是和左右直接相连

                AFT_LEN=0
                test_flag = 0


                numk = numk + 1


                # if numk==1032 or 1251:
                #     aoo=1
                # #
                #     node_num_display(Node_x_new, Node_y_new)
                if print_flag==1:
                    print("numk", numk)
                #
                # tnode1=3255
                # tnode2=3173
                # # if numk == 3089:
                # if numk < numk_flag:
                #     AFT_LEN = len(AFT_stack)
                #     for i in range(AFT_LEN):
                #         if AFT_stack[i][0] == tnode1 and AFT_stack[i][1] == tnode2:
                #             numk_flag = numk
                #     if numk_flag != 1000:
                #         jik = 1
                #
                # if numk > numk_flag:
                #     AFT_LEN = len(AFT_stack)
                #     for i in range(AFT_LEN):
                #         if AFT_stack[i][0] == tnode1 and AFT_stack[i][1] == tnode2:
                #             test_flag = 1
                #     if test_flag == 0:  #没有被改为1，表示这个阵面已经被删除
                #         node_num_display(Node_x_new, Node_y_new)
                #         print("numk", numk)
                #         jik = 1




                if numk>1:
                    AFT_stack.remove(AFT_stack[0])  #删除上一次的最短阵面，放到前面方便调试
                AFT_stack = sorted(AFT_stack, key=lambda fff: fff[4])  # 按照第5个元素重新排序，让最短的阵面在最上方

                if numk >=erro_test or len(AFT_stack)==0 :
                    time_e=time.time()
                    print("numk", numk)
                    print('time cost0', time_e - time_s, 's')

                    print("model1",model_num1)
                    print("model2", model_num2)
                    print("model3", model_num3)
                    print("model4", model_num4)
                    print("model5", model_num5)
                    print("model6", model_num6)
                    print("model7", model_num7)
                    print("model8", model_num8)
                    print("model9", model_num9)
                    print("model10", model_num10)
                    print("face_nun",face_num)
                    print("cell_nun", cell_num)
                    display()
                    node_num_display(Node_x_new, Node_y_new)


                    for i in  range(len(Left_Cell_Index_new)):
                        if Left_Cell_Index_new[i]==-1:
                            test_node.append([])
                            test_node[len(test_node)-1].append(Face_Node_Index_new[2*i])
                            test_node[len(test_node) - 1].append(Face_Node_Index_new[2 * i+1])
                            test_node[len(test_node)-1].append(Node_x_new[Face_Node_Index_new[2*i]-1])
                            test_node[len(test_node) - 1].append(Node_y_new[Face_Node_Index_new[2 * i] - 1])
                            test_node[len(test_node)-1].append(Node_x_new[Face_Node_Index_new[2*i+1]-1])
                            test_node[len(test_node) - 1].append(Node_y_new[Face_Node_Index_new[2 * i+1] - 1])

                    break

                node1 = AFT_stack[0][0]
                node2 = AFT_stack[0][1]
                # if node1==455 and node2==456:
                #     opp=1
                #     display()  # 显示边界
                #     node_num_display(Node_x_new, Node_y_new)
                node12_len = math.sqrt(
                    ((Node_x_new[node1 - 1] - Node_x_new[node2 - 1]) ** 2) + ((Node_y_new[node1 - 1] - Node_y_new[node2 - 1]) ** 2))
                node2_xy_1 = Translation(Node_x_new[node2 - 1], Node_y_new[node2 - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])
                m_flag=1.5
                ref_node = [0.5, (3 ** 0.5 / 2)*m_flag]  # 注意这里/除2是浮点型除法，整数除法是// two,往外推点，防止突发情况
                ref_node = Anti_Rotation(node2_xy_1[0], node2_xy_1[1], ref_node[0], ref_node[1])
                ref_node = Anti_Scaling(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node2 - 1], Node_y_new[node2 - 1],
                                        ref_node[0], ref_node[1])
                ref_node = Anti_Translation(ref_node[0], ref_node[1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])

                node0 = node1_nearby_adv(AFT_stack[0][0],AFT_stack[0][5],ref_node,node1,node2)[0]  # 取第一个阵面，找其函数返回两个值，取第一个

                node3 = node2_nearby_adv(AFT_stack[0][1],AFT_stack[0][5],ref_node,node1,node2)[0]
                mid_aft_x = (Node_x_new[node1 - 1] + Node_x_new[node2 - 1]) / 2  # 阵面中心x，y坐标
                mid_aft_y = (Node_y_new[node1 - 1] + Node_y_new[node2 - 1]) / 2

                real_step=step2(mid_aft_x,mid_aft_y)
                if real_step >= AFT_stack_out[0][4]:  # 返回的是该地方理想情况下下的阵面长度
                    real_step = AFT_stack_out[0][4]

                if node3==[] or node0==[]:
                    # display()  # 显示边界
                    node_num_display(Node_x_new, Node_y_new)


                ref_node3 = tangential_mid_point(Node_x_new[node2 - 1],Node_y_new[node2 - 1],Node_x_new[node3 - 1],Node_y_new[node3 - 1],real_step)
                node4=node2_nearby_adv(node3,-1,ref_node3,node2,node3)[0]  #-1代表编号，因为函数里面没有用，所以这里写个-1占位
                if node4==[] :
                    node_num_display(Node_x_new, Node_y_new)
                node_num=len(Node_x_new)

                two_ponit_len=math.sqrt(((Node_x_new[node1-1] - Node_x_new[node2-1]) ** 2) + ((Node_y_new[node1-1] - Node_y_new[node2-1]) ** 2))
                # if numk==2133:
                #     node_num_display(Node_x_new, Node_y_new)
                if node0==node3:
                    node_num_display(Node_x_new, Node_y_new)

                time_nn_s=time.clock()

                node0_xy = Translation(Node_x_new[node0 - 1], Node_y_new[node0 - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])
                node0_xy = Scaling(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node2 - 1], Node_y_new[node2 - 1],
                                   node0_xy[0], node0_xy[1])
                node0_xy = Rotation(node2_xy_1[0], node2_xy_1[1], node0_xy[0], node0_xy[1])

                node1_xy = Translation(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])
                node1_xy = Scaling(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node2 - 1], Node_y_new[node2 - 1],
                                   node1_xy[0], node1_xy[1])
                node1_xy = Rotation(node2_xy_1[0], node2_xy_1[1], node1_xy[0], node1_xy[1])

                node2_xy = Translation(Node_x_new[node2 - 1], Node_y_new[node2 - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])
                node2_xy = Scaling(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node2 - 1], Node_y_new[node2 - 1],
                                   node2_xy[0], node2_xy[1])
                node2_xy = Rotation(node2_xy_1[0], node2_xy_1[1], node2_xy[0], node2_xy[1])

                node3_xy = Translation(Node_x_new[node3 - 1], Node_y_new[node3 - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])
                node3_xy = Scaling(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node2 - 1], Node_y_new[node2 - 1],
                                   node3_xy[0], node3_xy[1])
                node3_xy = Rotation(node2_xy_1[0], node2_xy_1[1], node3_xy[0], node3_xy[1])


                node4_xy = Translation(Node_x_new[node4 - 1], Node_y_new[node4 - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])
                node4_xy = Scaling(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node2 - 1], Node_y_new[node2 - 1],
                                   node4_xy[0], node4_xy[1])
                node4_xy = Rotation(node2_xy_1[0], node2_xy_1[1], node4_xy[0], node4_xy[1])


                test_data_nn = [[node0_xy[0], node0_xy[1], node1_xy[0], node1_xy[1], node2_xy[0], node2_xy[1], node3_xy[0], node3_xy[1], node4_xy[0], node4_xy[1]]]
                #####################################################################
                final_output1 = sess.run(final_output, feed_dict={x_data: test_data_nn})

                #神经网络##########################################
                ######################################################################
                time_end2 = time.clock()

                output2 = final_output1[0]
                aaa = output2.tolist()
                aaa_max = aaa.index(max(aaa))
                model = aaa_max + 1  # 实际的加1才是真实的

                time_nn_e = time.clock()
                # print("cnn time",time_nn_e-time_nn_s)

                time_end3 = time.time()


                node1_nearby=node1_nearby_adv(AFT_stack[0][0], AFT_stack[0][5],ref_node,node1,node2)  #包含两个原变量，一个是点，和face的编号
                node2_nearby=node2_nearby_adv(AFT_stack[0][1], AFT_stack[0][5],ref_node,node1,node2)
                ccc = node1_nearby[0]
                first_face_num=node1_nearby[1]
                ddd = node2_nearby[0]
                third_face_num=node2_nearby[1]

                node2_xy_1 = Translation(Node_x_new[node2 - 1], Node_y_new[node2 - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1])

                ref_node = [0.5, (3 ** 0.5 / 2)*m_flag]  # 注意这里/除2是浮点型除法，整数除法是// two
                node2_xy_2 = Translation(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[ccc - 1], Node_y_new[ccc - 1])   #左边face的标准，以便恢复所用
                ref_node = Anti_Rotation(node2_xy_1[0], node2_xy_1[1], ref_node[0], ref_node[1])
                ref_node = Anti_Scaling(Node_x_new[ccc - 1], Node_y_new[ccc - 1], Node_x_new[node1 - 1], Node_y_new[node1 - 1], ref_node[0], ref_node[1])#比例尺寸就是ccc到node1的长度
                ref_node = Anti_Translation(ref_node[0], ref_node[1], Node_x_new[ccc - 1], Node_y_new[ccc - 1])  #ccc是起点，圆点参考点

                node1_node1_nearby=node1_nearby_adv(ccc,node1_nearby[1],ref_node,node0,node1)
                eee=node1_node1_nearby[0]

                ref_node2 = [0.5, (3 ** 0.5 / 2)*m_flag]  # 注意这里/除2是浮点型除法，整数除法是// two   求ddd到node2这face的中心，为了找到fff
                node2_xy_3 = Translation(Node_x_new[ddd - 1], Node_y_new[ddd - 1], Node_x_new[node2 - 1],
                                         Node_y_new[node2 - 1])  # 左边face的标准，以便恢复所用
                ref_node2 = Anti_Rotation(node2_xy_3[0], node2_xy_3[1], ref_node2[0], ref_node2[1])
                ref_node2 = Anti_Scaling(Node_x_new[ddd - 1], Node_y_new[ddd - 1], Node_x_new[node2 - 1], Node_y_new[node2 - 1],
                                        ref_node2[0], ref_node2[1])  # 比例尺寸就是ddd到node2的长度
                ref_node2 = Anti_Translation(ref_node2[0], ref_node2[1], Node_x_new[ccc - 1], Node_y_new[ccc - 1])  # ccc是起点，圆点参考点

                node2_node2_nearby = node2_nearby_adv(ddd, node2_nearby[1], ref_node2,node2,node3)
                fff = node2_node2_nearby[0]
                four_face_num=node2_node2_nearby[1]



                new_node1 = tangential_mid_point(Node_x_new[node1 - 1], Node_y_new[node1 - 1], Node_x_new[node2 - 1],
                                                 Node_y_new[node2 - 1], real_step)

                # 新生成的点反归一化##def Translation(x,y,x0,y0):#x,y表示要任何要平移的点，x0，y0表示将以何点为圆心作为参考点
                if first_face_num==656 or third_face_num==656 or four_face_num==656:
                    ooo=1

                new_node2 = ref_node3
                len_new_node1_node2=face_len(new_node1[0],new_node1[1],new_node2[0],new_node2[1])
                if model==1: #防止模式1两个新点太近的情况，通过专家系统转化为模式5
                    if len_new_node1_node2<0.7*two_ponit_len:
                        model=5
                # if node1==61 and node2==60:
                #     model =1
                #
                # if node1==1773 and node2==2072:
                #     model =6
                #     node_num_display(Node_x_new, Node_y_new)

                # if node1==1848 and node2==1971:
                #     node_num_display(Node_x_new, Node_y_new)
                if node0==node4:   #对于0与4相等的情况，经常判断不好，直接写成成死程序就可以了，可能质量不好，可以后面优化，8 和10本质上没有区别
                    jiaodu = vetor_angle(node1, node2, node3)
                    if jiaodu <= 90:
                        model = 10
                    else:
                        model = 8
                if print_flag==1:

                    print(node0,node1,node2,node3,node4,len(Node_x_new))
                if model==6:

                    print ("error model",model)
                    print("阵面左边点，阵面右边点"), AFT_stack[0][0], AFT_stack[0][1]
                    print("新点xy", Node_x_new[AFT_stack[0][0] - 1], Node_y_new[AFT_stack[0][0] - 1])
                    # display()  # 显示边界
                    node_num_display(Node_x_new, Node_y_new)

                    node0_xy = unit_direction_add(node1_xy, node0_xy)  # the first paramiter is start point of vetor
                    node3_xy_temp = unit_direction_add(node2_xy, node3_xy)  #
                    node4_unit=unit_direction(node3_xy, node4_xy)  #
                    node3_xy=node3_xy_temp
                    node4_xy[0]=node3_xy[0]+node4_unit[0]
                    node4_xy[1] = node3_xy[1] + node4_unit[1]


                    test_data_nn = [
                        [node0_xy[0], node0_xy[1], node1_xy[0], node1_xy[1], node2_xy[0], node2_xy[1], node3_xy[0],
                         node3_xy[1], node4_xy[0], node4_xy[1]]]
                    #####################################################################
                    final_output1 = sess.run(final_output, feed_dict={x_data: test_data_nn})
                    output2 = final_output1[0]
                    aaa = output2.tolist()
                    aaa_max = aaa.index(max(aaa))
                    model = aaa_max + 1  # 实际的加1才是真实的
                    print("修正后的 model", model)

                if model==10 or model==8:
                    if node0 !=node4:
                        print ("error model",model)
                        print("阵面左边点，阵面右边点"), AFT_stack[0][0], AFT_stack[0][1]
                        print("新点xy", Node_x_new[AFT_stack[0][0] - 1], Node_y_new[AFT_stack[0][0] - 1])
                        # display()  # 显示边界
                        node_num_display(Node_x_new, Node_y_new)

                        node0_xy = unit_direction_add(node1_xy, node0_xy)  # the first paramiter is start point of vetor
                        node3_xy_temp = unit_direction_add(node2_xy, node3_xy)  #
                        node4_unit=unit_direction(node3_xy, node4_xy)  #
                        node3_xy=node3_xy_temp
                        node4_xy[0]=node3_xy[0]+node4_unit[0]
                        node4_xy[1] = node3_xy[1] + node4_unit[1]

                        # node0_node2_x = [node0_xy[0], node1_xy[0]]
                        # node0_node2_y = [node0_xy[1], node1_xy[1]]
                        # plt.plot(node0_node2_x, node0_node2_y, 'g-s', linewidth=3, color='r', markerfacecolor='r',
                        #          marker='o')
                        #
                        # node0_node2_x = [node1_xy[0], node2_xy[0]]
                        # node0_node2_y = [node1_xy[1], node2_xy[1]]
                        # plt.plot(node0_node2_x, node0_node2_y, 'g-s', linewidth=3, color='r', markerfacecolor='r',
                        #          marker='o')
                        #
                        # node0_node2_x = [node2_xy[0], node3_xy[0]]
                        # node0_node2_y = [node2_xy[1], node3_xy[1]]
                        # plt.plot(node0_node2_x, node0_node2_y, 'g-s', linewidth=3, color='r', markerfacecolor='r',
                        #          marker='o')
                        #
                        # node0_node2_x = [node3_xy[0], node4_xy[0]]
                        # node0_node2_y = [node3_xy[1], node4_xy[1]]
                        # plt.plot(node0_node2_x, node0_node2_y, 'g-s', linewidth=3, color='r', markerfacecolor='r',
                        #          marker='o')

                        test_data_nn = [
                            [node0_xy[0], node0_xy[1], node1_xy[0], node1_xy[1], node2_xy[0], node2_xy[1], node3_xy[0],
                             node3_xy[1], node4_xy[0], node4_xy[1]]]
                        #####################################################################
                        final_output1 = sess.run(final_output, feed_dict={x_data: test_data_nn})
                        output2 = final_output1[0]
                        aaa = output2.tolist()
                        aaa_max = aaa.index(max(aaa))
                        model = aaa_max + 1  # 实际的加1才是真实的

                if model==8:
                    if node0 !=node4:
                        print ("error model",model)
                        model=model-1

                if model == 10:
                    if node0 != node4:
                        print ("error model",model)
                        jiaodu=vetor_angle(node1,node2,node3)
                        if jiaodu<=90:
                            model=9
                        elif jiaodu>180:
                            model = 1
                        else:
                            model = 5

                if node1 == 30 and node2==29:
                    model=7






                # if model==4 and node0==node4:
                #     model=8

                if model == 1:  #
                    model_num1=model_num1+1

                    if print_flag==1:
                        print("model=",model)
                    cell_num = cell_num + 1  # 多一个cell
                    # 完善以前face的左边cell
                    Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元

                    New_node_nearby=[]  #清空存放相邻点的二维list
                    New_face_nearby=[]
                    Node_x_new.append(new_node1[0])
                    Node_y_new.append(new_node1[1])

                    node_nearby(new_node1,two_ponit_len,node1,node2)#查周围点的函数，放在New_node_nearby中
                    cross_judge1=cross_judge(node1,node2)
                    cross_flag_1=cross_judge1[0]
                    Pbest_num = cross_judge1[1]
                    new_node1_temp=Pbest_num[0]  #将新点1暂存，后面备用
                    new_node_num1=len(Node_x_new)
                    flag_eee=0
                    near_node_flag=0 #推出去两个新点都在阵面上，而且相邻的情况，这种在内场到边界的时候最为常见，每次都归零
                    near_node_flag = Pbest_num[0]
                    if cross_flag_1 == 0:  # nearnode1_nearby_adv
                        if ccc==Pbest_num[0]: #看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                            add_face(node0,node2,-1,cell_num)
                            add_AFT_stack(node0,node2,face_num,cell_num)
                            display_face(node0,node2)
                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                        elif eee==Pbest_num[0]: #看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # cell_num = cell_num + 1  # 多一个cell  0417
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            add_face(node1, Pbest_num[0],cell_num+1, cell_num)
                            display_face(node1, Pbest_num[0])

                            add_face(Pbest_num[0],node2,-1,cell_num)
                            add_AFT_stack(Pbest_num[0],node2,face_num,cell_num)
                            display_face(Pbest_num[0],node2)

                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num  # 补充阵面的左单元


                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])
                            flag_eee=1


                        elif Pbest_num[0] <=len(Node_x_new)-1 : #判断最优点是不是在阵面上，如果在，直接连接，如果不在，则直接走到else中去
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # cell_num = cell_num + 1  # 多一个cell  0417
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元

                            add_face(node1,Pbest_num[0],-1,cell_num)
                            add_AFT_stack(node1,Pbest_num[0],face_num,cell_num)
                            display_face(node1,Pbest_num[0])

                            add_face(Pbest_num[0],node2,-1,cell_num)
                            add_AFT_stack(Pbest_num[0],node2,face_num,cell_num)
                            display_face(Pbest_num[0],node2)
                            near_node_flag = Pbest_num[0]

                        else:
                            add_face(node1, new_node_num1, -1, cell_num)
                            add_AFT_stack(node1, new_node_num1, face_num, cell_num)
                            display_face(node1, new_node_num1)

                            add_face(new_node_num1,node2,  -1, cell_num)
                            add_AFT_stack(new_node_num1,node2,  face_num, cell_num)
                            display_face(new_node_num1,node2)


                    cell_num = cell_num + 1  # 多一个cell
                    # 完善以前face的左边cell
                    Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元

                    New_node_nearby = []  # 清空存放相邻点的二维list
                    New_face_nearby = []
                    Node_x_new.append(new_node2[0])
                    Node_y_new.append(new_node2[1])

                    node_nearby(new_node2, two_ponit_len,node2,node3) #这里暂时用two_ponit_len，本质上应该用node2 node3的阵面长度
                    cross_judge2=cross_judge(node2,node3)
                    cross_flag_2=cross_judge2[0]
                    Pbest_num = cross_judge2[1]
                    new_node_num2=len(Node_x_new)

                    if cross_flag_2 == 0:  # nearnode1_nearby_adv

                        node1_node1_node1_nearby = node1_nearby_adv(node1_node1_nearby[0], node1_node1_nearby[1], ref_node,eee,ccc)
                        ggg=node1_node1_node1_nearby[0]
                        node2_node2_node2_nearby = node2_nearby_adv(node2_node2_nearby[0], node2_node2_nearby[1], ref_node2,ddd,fff)
                        hhh=node2_node2_node2_nearby[0]

                        if fff==Pbest_num[0] :   #看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            if new_node1_temp != hhh:

                                Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号
                                add_face(node2,node4,-1,cell_num)
                                display_face(node2, node4)

                                add_AFT_stack(node2,node4,face_num,cell_num)
                                AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                            else:

                                aft_len=len(AFT_stack)
                                AFT_stack.remove(AFT_stack[aft_len-1])  #最后一个阵面删除
                                Left_Cell_Index_new[face_num - 1] = cell_num+1 #补充上次生成的最后一条边的左cell   1

                                Left_Cell_Index_new[ four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号    2
                                add_face(node2, node4, cell_num +1, cell_num)  # 12
                                display_face(node2, node4)
                                AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)

                                cell_num=cell_num+1
                                face_temp=node2_node2_node2_nearby[1]
                                Left_Cell_Index_new[face_temp - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号
                                AFT_stack = del_adv_elemnt(AFT_stack, face_temp)

                        elif new_node1_temp==Pbest_num[0] : #两个点是同一个点，本来是生成两个点，可能由于选择了现有的点，所以
                            if hhh!=Pbest_num[0]:
                                Node_x_new.pop()
                                Node_y_new.pop()
                                AFT_stack = del_adv_elemnt(AFT_stack, face_num) #删除最后一条边所生成的阵面


                                add_face(Pbest_num[0],node3, -1, cell_num)
                                display_face(Pbest_num[0],node3)
                                add_AFT_stack(Pbest_num[0],node3, face_num, cell_num)
                            else:
                                Node_x_new.pop()
                                Node_y_new.pop()
                                Left_Cell_Index_new[face_num - 1] = cell_num  #最后一条边补充阵面
                                AFT_stack = del_adv_elemnt(AFT_stack, face_num)  # 删除最后一条边所生成的阵面

                                Left_Cell_Index_new[third_face_num - 1] = cell_num  # 最后一条边补充阵面

                                add_face(Pbest_num[0], node3, cell_num+1, cell_num)
                                display_face(Pbest_num[0], node3)

                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[four_face_num - 1] = cell_num  # 最后一条边补充阵面
                                AFT_stack = del_adv_elemnt(AFT_stack, four_face_num )  # 删除最后一条边所生成的阵面

                                Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  # 补充阵面的左单元
                                AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])




                        elif hhh==Pbest_num[0]: #看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            oneaft_flag=is_in_oneaft(Pbest_num[0], new_node1_temp)  #返回值第一位是是否在一个阵面上，2位是
                            if oneaft_flag[0]==0:
                                # cell_num = cell_num + 1  # 多一个cell   0417
                                Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                                add_face(node2, Pbest_num[0],-1, cell_num)
                                display_face(node2, Pbest_num[0])
                                add_AFT_stack(node2, Pbest_num[0], face_num, cell_num)

                                add_face(Pbest_num[0],node3,cell_num+1,cell_num)
                                display_face(Pbest_num[0],node3)

                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                                Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                                # AFT_stack = del_adv_elemnt(AFT_stack, third_face_num) #后面统一删除
                                AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                                AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])
                            else:
                                # cell_num = cell_num + 1  # 多一个cell   0417
                                Left_Cell_Index_new[face_num - 1] = cell_num+2  # 补充阵面的左单元
                                AFT_stack = del_adv_elemnt(AFT_stack, face_num)   #把第一个点生成的最后一个边形成的阵面去掉，因为会生成一个封闭的三角形，
                                Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                                add_face(node2, Pbest_num[0], cell_num+2, cell_num)
                                display_face(node2, Pbest_num[0])

                                add_face(Pbest_num[0], node3, cell_num + 1, cell_num)
                                display_face(Pbest_num[0], node3)

                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                                Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                                # AFT_stack = del_adv_elemnt(AFT_stack, third_face_num) #后面统一删除
                                AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                                AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])

                                cell_num = cell_num + 1  # 多一个cell
                                AFT_stack = del_adv_elemnt(AFT_stack, oneaft_flag[1])

                        elif ggg == Pbest_num[0]:  # 看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            if flag_eee==1:
                                # node_num_display(Node_x_new, Node_y_new)
                                Node_x_new.pop()
                                Node_y_new.pop()
                                # cell_num = cell_num + 1  # 多一个cell  0417
                                Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                                Left_Cell_Index_new[face_num - 1] = cell_num+1  # 补充最后一条边的左单元号

                                AFT_stack = del_adv_elemnt(AFT_stack, face_num) #把最后一条边所形成的阵面删除
                                add_face(node2, Pbest_num[0], cell_num+1, cell_num)
                                display_face(node2, Pbest_num[0])

                                add_face(Pbest_num[0],node3, cell_num+1, cell_num)
                                display_face(Pbest_num[0],node3)
                                add_AFT_stack(Pbest_num[0],node3, face_num, cell_num)

                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[node1_node1_node1_nearby[1] - 1] = cell_num  # 补充最后一条边的左单元号
                                AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_node1_nearby[1])  # 以前的阵面删除



                        elif Pbest_num[0] <=len(Node_x_new)-2 : #判断最优点是不是在阵面上，如果在，直接连接，如果不在，则直接走到else中去,因为加第二个点了，所以要减去2，这个点不能是本次生成的第一个点
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # cell_num = cell_num + 1  # 多一个cell  0417
                            # Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            near_flag=0
                            face_num_temp=0
                            for i in range(len(AFT_stack)):
                                if (AFT_stack[i][0]==near_node_flag and AFT_stack[i][1]==Pbest_num[0]) or (AFT_stack[i][1]==near_node_flag and AFT_stack[i][0]==Pbest_num[0]): #检测两个点是不是在同一个阵面上
                                    near_flag = 1
                                    face_num_temp = AFT_stack[i][5]   #记录阵面，后面删除
                                    break

                            if near_flag ==1:#表示第一个阵面长出去的点再已有的阵面上，而且与第二阵面所长出去的点（也在阵面上）相邻，则会多构成一个三角形
                                Left_Cell_Index_new[face_num - 1] = cell_num + 1  # 补充最后一条边的左单元号，这里加1是因为是多了一个cell
                                AFT_stack = del_adv_elemnt(AFT_stack, face_num)  # 把最后一条边所形成的阵面删除
                                add_face(node2, Pbest_num[0], cell_num + 1, cell_num)

                            else:
                                add_face(node2, Pbest_num[0], -1, cell_num)
                                add_AFT_stack(node2,Pbest_num[0],face_num,cell_num)

                            display_face(node2,Pbest_num[0])

                            add_face(Pbest_num[0],node3,-1,cell_num)
                            add_AFT_stack(Pbest_num[0],node3,face_num,cell_num)
                            display_face(Pbest_num[0],node3)
                            if near_flag == 1:  # 表示第一个阵面长出去的点再已有的阵面上，而且与第二阵面所长出去
                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[face_num_temp - 1] = cell_num  # 补充阵面的左单元
                                AFT_stack = del_adv_elemnt(AFT_stack, face_num_temp)

                        else:
                            add_face(node2, new_node_num2, -1, cell_num)
                            display_face(node2, new_node_num2)
                            add_AFT_stack(node2, new_node_num2, face_num, cell_num)
                            add_face(new_node_num2,node3, -1, cell_num)
                            display_face(new_node_num2,node3)
                            add_AFT_stack(new_node_num2,node3, face_num, cell_num)
                    AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                    time_m1_e = time.clock()
                    # print("模式1时间",time_m1_e-time_nn_s)
                elif model == 2:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    # 第二个阵面生成的新点，第一个阵面不生成新点
                    model_num2 = model_num2 + 1
                    if print_flag==1:
                        print("model=", model)
                    #第一个三角形
                    cell_num = cell_num + 1  # 多一个cell
                    Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                    Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元

                    add_face(node0, node2, -1, cell_num)
                    add_AFT_stack(node0, node2, face_num, cell_num)
                    display_face(node0, node2)
                    AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)

                    # 第二个三角形
                    cell_num = cell_num + 1  # 多一个cell
                    # 完善以前face的左边cell
                    Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元

                    New_node_nearby = []  # 清空存放相邻点的二维list
                    New_face_nearby = []
                    Node_x_new.append(new_node2[0])
                    Node_y_new.append(new_node2[1])

                    node_nearby(new_node2, two_ponit_len,node2,node3) #这里暂时用two_ponit_len，本质上应该用node2 node3的阵面长度
                    cross_judge2=cross_judge(node2,node3)
                    cross_flag_2=cross_judge2[0]
                    Pbest_num = cross_judge2[1]
                    new_node_num2=len(Node_x_new)

                    if cross_flag_2 == 0:  # nearnode1_nearby_adv
                        node2_node2_node2_nearby = node2_nearby_adv(node2_node2_nearby[0], node2_node2_nearby[1], ref_node2,ddd,fff)
                        hhh=node2_node2_node2_nearby[0]
                        if ccc==Pbest_num[0] and  hhh !=Pbest_num[0]:   #看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            AFT_stack = del_adv_elemnt(AFT_stack, face_num) #把刚才上一个三角形最后一个阵面删除了，因为这构成闭合的了，
                            Left_Cell_Index_new[face_num- 1] = cell_num  # 最后一条边的左阵面补充完整
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  #
                            add_face(node0, node3, -1, cell_num)
                            display_face(node0, node3)
                            add_AFT_stack(node0, node3, face_num, cell_num)

                        elif ccc==Pbest_num[0] and  hhh ==Pbest_num[0]:  # 看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            AFT_stack = del_adv_elemnt(AFT_stack, face_num)  # 把刚才上一个三角形最后一个阵面删除了
                            Left_Cell_Index_new[face_num - 1] = cell_num  # 最后一条边的左阵面补充完整,根据图形需要加1
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  #
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)  # 把刚才上一个三角形最后一个阵面删除了
                            add_face(Pbest_num[0], node3, cell_num+1, cell_num)
                            display_face(Pbest_num[0], node3)

                            # 第3个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[node1_node1_nearby[1]- 1] = cell_num  #
                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])  # 把刚才上一个三角形最后一个阵面删除了

                            Left_Cell_Index_new[four_face_num- 1] = cell_num  #
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)  #

                        elif eee == Pbest_num[0] and hhh !=Pbest_num[0]:  # 看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            AFT_stack = del_adv_elemnt(AFT_stack, face_num)  # 把刚才上一个三角形最后一个阵面删除了
                            Left_Cell_Index_new[face_num - 1] = cell_num+1  # 最后一条边的左阵面补充完整,根据图形需要加1
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  #
                            add_face(node2, Pbest_num[0], cell_num+1, cell_num)
                            display_face(node2, Pbest_num[0])

                            add_face(Pbest_num[0], node3, -1, cell_num)
                            display_face(Pbest_num[0], node3)
                            add_AFT_stack(Pbest_num[0], node3, face_num, cell_num)

                            # 第3个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[node1_node1_nearby[1]- 1] = cell_num  #
                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])  # 把刚才上一个三角形最后一个阵面删除了

                        elif eee == Pbest_num[0] and hhh ==Pbest_num[0]:  # 看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            AFT_stack = del_adv_elemnt(AFT_stack, face_num)  # 把刚才上一个三角形最后一个阵面删除了
                            Left_Cell_Index_new[face_num - 1] = cell_num+1  # 最后一条边的左阵面补充完整,根据图形需要加1
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  #
                            add_face(node2, Pbest_num[0], cell_num+1, cell_num)
                            display_face(node2, Pbest_num[0])

                            add_face(Pbest_num[0], node3, cell_num+2, cell_num)
                            display_face(Pbest_num[0], node3)

                            # 第3个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[node1_node1_nearby[1]- 1] = cell_num  #
                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])  # 把刚才上一个三角形最后一个阵面删除了

                            # 第4个三角形

                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  #
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)  # 把刚才上一个三角形最后一个阵面删除了

                            Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  #
                            AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])  # 把刚才上一个三角形最后一个阵面删除了


                        elif fff == Pbest_num[0]:  # 看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            Left_Cell_Index_new[
                                four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号
                            add_face(node2, node4, -1, cell_num)
                            display_face(node2, node4)
                            add_AFT_stack(node2, node4, face_num, cell_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)

                        elif hhh==Pbest_num[0]: #看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()

                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                            add_face(node2, Pbest_num[0],-1, cell_num)
                            display_face(node2, Pbest_num[0])
                            add_AFT_stack(node2, Pbest_num[0], face_num, cell_num)

                            add_face(Pbest_num[0],node3,cell_num+1,cell_num)
                            display_face(Pbest_num[0],node3)

                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])
                        elif Pbest_num[0] <=len(Node_x_new)-2 : #判断最优点是不是在阵面上，如果在，直接连接，如果不在，则直接走到else中去,因为加第二个点了，所以要减去2，这个点不能是本次生成的第一个点
                            Node_x_new.pop()
                            Node_y_new.pop()
                            node_num_display(Node_x_new, Node_y_new)
                            add_face(node2, Pbest_num[0], -1, cell_num)
                            display_face(node2, Pbest_num[0])
                            add_AFT_stack(node2, Pbest_num[0],face_num, cell_num)
                            add_face(Pbest_num[0],node3,  -1, cell_num)
                            display_face(Pbest_num[0],node3)
                            add_AFT_stack(Pbest_num[0],node3,  face_num, cell_num)
                        else:
                            add_face(node2, new_node_num2, -1, cell_num)
                            display_face(node2, new_node_num2)
                            add_AFT_stack(node2, new_node_num2,face_num, cell_num)
                            add_face(new_node_num2,node3,  -1, cell_num)
                            display_face(new_node_num2,node3)
                            add_AFT_stack(new_node_num2,node3,  face_num, cell_num)
                    AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                    time_m2_e = time.clock()
                    # print("模式2时间",time_m2_e-time_nn_s)

                elif model == 3:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    model_num3 = model_num3 + 1
                    if print_flag==1:
                        print("model=", model)
                    ##############################################以下是两个新点的位置############################################
                    cell_num = cell_num + 1  # 多一个cell
                    # 完善以前face的左边cell
                    Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元

                    New_node_nearby = []  # 清空存放相邻点的二维list
                    New_face_nearby = []
                    Node_x_new.append(new_node1[0])
                    Node_y_new.append(new_node1[1])

                    node_nearby(new_node1, two_ponit_len,node1, node2)  # 查周围点的函数，放在New_node_nearby中
                    cross_judge1 = cross_judge(node1, node2)
                    cross_flag_1 = cross_judge1[0]
                    Pbest_num = cross_judge1[1]
                    new_node_num1 = len(Node_x_new)
                    # node1_node1_node1_nearby = node1_nearby_adv(node1_node1_nearby[0], node1_node1_nearby[1], ref_node, eee, ccc)
                    # ggg = node1_node1_node1_nearby[0]
                    node2_node2_node2_nearby = node2_nearby_adv(node2_node2_nearby[0], node2_node2_nearby[1], ref_node2, ddd, fff)
                    hhh = node2_node2_node2_nearby[0]


                    if cross_flag_1 == 0:  # nearnode1_nearby_adv
                        if ccc == Pbest_num[0]:  # 看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()
                            Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                            add_face(node0, node2, -1, cell_num)
                            add_AFT_stack(node0, node2, face_num, cell_num)
                            display_face(node0, node2)
                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                        elif eee==Pbest_num[0] and fff!=Pbest_num[0]: #看新点是不是阵面左点的临近点，如果是，则最佳点不是新生成的点，这时新生成的点还没有构建face数据结构
                            Node_x_new.pop()
                            Node_y_new.pop()

                            add_face(node1, Pbest_num[0],cell_num+1, cell_num)
                            display_face(node1, Pbest_num[0])

                            add_face(Pbest_num[0],node2,-1,cell_num)
                            add_AFT_stack(Pbest_num[0],node2,face_num,cell_num)
                            display_face(Pbest_num[0],node2)

                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])

                        elif eee!=Pbest_num[0] and fff==Pbest_num[0]: #看新点是不是阵成
                            Node_x_new.pop()
                            Node_y_new.pop()

                            add_face(node1, Pbest_num[0],-1, cell_num)
                            display_face(node1, Pbest_num[0])
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)

                            add_face(Pbest_num[0],node2,cell_num+1,cell_num)   #这种情况会生成三个三角形，所以看图
                            display_face(Pbest_num[0],node2)

                            cell_num = cell_num + 1  # 第2个cell
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号

                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                        elif eee==Pbest_num[0] and fff==Pbest_num[0]: #看新点是不是阵成
                            Node_x_new.pop()
                            Node_y_new.pop()

                            add_face(node1, Pbest_num[0],cell_num+1, cell_num)
                            display_face(node1, Pbest_num[0])

                            add_face(Pbest_num[0],node2,cell_num+2,cell_num)   #这种情况会生成三个三角形，所以看图
                            display_face(Pbest_num[0],node2)

                            cell_num = cell_num + 1  # 第2个cell
                            Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num

                            cell_num = cell_num + 1  # 第2个cell
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号


                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)

                        elif Pbest_num[0] <=len(Node_x_new)-1 : #判断最优点是不是在阵面上，如果在，直接连接，如果不在，则直接走到else中去
                            Node_x_new.pop()
                            Node_y_new.pop()
                            add_face(node1, Pbest_num[0], -1, cell_num)
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)
                            display_face(node1, Pbest_num[0])

                            add_face(Pbest_num[0], node2, -1, cell_num)
                            add_AFT_stack(Pbest_num[0], node2, face_num, cell_num)
                            display_face(Pbest_num[0], node2)

                        else:
                            add_face(node1, Pbest_num[0], -1, cell_num)
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)
                            display_face(node1, Pbest_num[0])

                            add_face(Pbest_num[0], node2, -1, cell_num)
                            add_AFT_stack(Pbest_num[0], node2, face_num, cell_num)
                            display_face(Pbest_num[0], node2)

                    # 第二个三角形
                    if fff!=Pbest_num[0]:
                        if hhh !=Pbest_num[0]:
                            cell_num = cell_num + 1  # 多一个cell
                            # 完善以前face的左边cell
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元

                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号
                            add_face(node2, node4, -1, cell_num)
                            display_face(node2, node4)
                            add_AFT_stack(node2, node4, face_num, cell_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                        else:
                            cell_num = cell_num + 1  # 多一个cell
                            # 完善以前face的左边cell
                            face_num_temp=AFT_stack[len(AFT_stack)-1][5]

                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号
                            add_face(node2, node4, cell_num+1, cell_num)
                            display_face(node2, node4)
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)

                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[face_num_temp - 1] = cell_num  # 补充阵面的左单元
                            AFT_stack = del_adv_elemnt(AFT_stack, face_num_temp)
                            AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])

                    time_m3_e = time.clock()
                    # print("模式3时间",time_m3_e-time_nn_s)

                elif model == 4:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    model_num4 = model_num4 + 1
                    if print_flag==1:
                        print("model=", model)
                    if eee!=node4:
                        #第一个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                        Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元

                        add_face(node0, node2, -1, cell_num)
                        add_AFT_stack(node0, node2, face_num, cell_num)
                        display_face(node0, node2)
                        AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)

                        # 第二个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        # 完善以前face的左边cell
                        Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元

                        Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号
                        add_face(node2, node4, -1, cell_num)
                        display_face(node2, node4)
                        add_AFT_stack(node2, node4, face_num, cell_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                    else:
                        #第一个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                        Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元

                        add_face(node0, node2, cell_num+1+1, cell_num)
                        display_face(node0, node2)
                        AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)

                        # 第二个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        # 完善以前face的左边cell
                        Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元

                        Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充node0-node1face的左单元，node1_nearby_adv函数返回两个值，第2个是face编号
                        add_face(node2, node4, cell_num+1, cell_num)
                        display_face(node2, node4)

                        # 第3个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                        AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])
                        AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                    time_m4_e = time.clock()
                    # print("模式4时间",time_m4_e-time_nn_s)




                elif model == 5:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    model_num5 = model_num5 + 1
                    if print_flag==1:
                        print("model=", model)

                    ################################y一下是新点的位置，有两个新点中和而成####################################
                    new_node_final=[0.5*(new_node1[0]+new_node2[0]),0.5*(new_node1[1]+new_node2[1])]

                    Node_x_new.append(new_node_final[0])
                    Node_y_new.append(new_node_final[1])
                    new_node_num1 = len(Node_x_new)
                    node_nearby(new_node_final, two_ponit_len, node1, node2)  # 查周围点的函数，放在New_node_nearby中
                    cross_judge1 = cross_judge(node1, node2)
                    cross_flag_1 = cross_judge1[0]
                    Pbest_num = cross_judge1[1]
                    new_node_num1 = len(Node_x_new)
                    flag_eee = 0
                    node2_node2_node2_nearby = node2_nearby_adv(node2_node2_nearby[0], node2_node2_nearby[1], ref_node2, ddd, fff)
                    hhh = node2_node2_node2_nearby[0]
                    if cross_flag_1 == 0:  # nearnode1_nearby_adv
                        if eee==Pbest_num[0] and fff != Pbest_num[0] and hhh != Pbest_num[0]:
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # 第一个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[first_face_num - 1] = cell_num+1  # 补充阵面的左单元，请见程序调试记录2021.4.14阵面460 610
                            Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num + 1

                            add_face(node1, Pbest_num[0], cell_num+1, cell_num)
                            display_face(node1, Pbest_num[0])
                            add_face(Pbest_num[0],node2,  cell_num+2, cell_num)  #lupeng 20210514
                            display_face(Pbest_num[0],node2)

                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])
                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                            # 第2个三角形
                            cell_num = cell_num + 1  # 多一个cell,事情处理完了，只加一下就行了
                            # 第3个三角形  610 611 603
                            cell_num = cell_num + 1
                            add_face(Pbest_num[0],node3,  -1, cell_num)
                            display_face(Pbest_num[0],node3)
                            add_AFT_stack(Pbest_num[0],node3, face_num, cell_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                        elif fff == Pbest_num[0] and eee==Pbest_num[0]:
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # 第一个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[first_face_num - 1] = cell_num+1  # 补充阵面的左单元，请见程序调试记录2021.4.14阵面460 610
                            Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num + 1

                            add_face(node1, Pbest_num[0], cell_num+1, cell_num)
                            display_face(node1, Pbest_num[0])
                            add_face(Pbest_num[0],node2,  cell_num+2, cell_num)  #lupeng 20210514
                            display_face(Pbest_num[0],node2)

                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])
                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                            # 第2个三角形
                            cell_num = cell_num + 1  # 多一个cell,事情处理完了，只加一下就行了
                            # 第3个三角形  610 611 603
                            cell_num = cell_num + 1
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)

                        elif hhh == Pbest_num[0] and eee == Pbest_num[0]:
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # 第一个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[first_face_num - 1] = cell_num+1  # 补充阵面的左单元，请见程序调试记录2021.4.14阵面460 610
                            Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num + 1

                            add_face(node1, Pbest_num[0], cell_num+1, cell_num)
                            display_face(node1, Pbest_num[0])
                            add_face(Pbest_num[0],node2,  cell_num+2, cell_num)  #lupeng 20210514
                            display_face(Pbest_num[0],node2)

                            AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])
                            AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                            # 第2个三角形
                            cell_num = cell_num + 1  # 多一个cell,事情处理完了，只加一下就行了
                            # 第3个三角形  610 611 603
                            cell_num = cell_num + 1
                            Left_Cell_Index_new[third_face_num- 1] = cell_num
                            add_face(Pbest_num[0],node3,  cell_num+1, cell_num)
                            display_face(Pbest_num[0],node3)
                            # 第4个三角形  610 611 603
                            cell_num = cell_num + 1
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])






                        elif fff == Pbest_num[0] and eee!=Pbest_num[0]:
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # 第一个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[third_face_num - 1] = cell_num+1  # 补充阵面的左单元
                            Left_Cell_Index_new[four_face_num - 1] = cell_num+1  # 补充阵面的左单元

                            add_face(node1, Pbest_num[0], -1, cell_num)
                            display_face(node1, Pbest_num[0])
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)
                            add_face( Pbest_num[0],node2, cell_num+1, cell_num)
                            display_face(Pbest_num[0],node2)
                            cell_num = cell_num + 1  # 多一个cell  事情前面处理完了
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                        elif ddd == Pbest_num[0]:
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # 第一个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                            add_face(node1, Pbest_num[0], -1, cell_num)
                            display_face(node1, Pbest_num[0])
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)


                        elif ccc == Pbest_num[0]:
                            if eee!=fff:
                                Node_x_new.pop()
                                Node_y_new.pop()
                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                                Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元

                                add_face(node0, node2, cell_num + 1, cell_num)
                                display_face(node0, node2)
                                AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)

                                # 第二个三角形
                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                                add_face(node0, node3, -1, cell_num)  # 这条边不会生成阵面
                                add_AFT_stack(node0, node3, face_num, cell_num)
                                display_face(node0, node3)
                                AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            else:
                                Node_x_new.pop()
                                Node_y_new.pop()

                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                                Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元

                                add_face(node0, node2, cell_num + 1, cell_num)
                                display_face(node0, node2)
                                AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)

                                # 第二个三角形
                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                                add_face(node0, node3, cell_num+1, cell_num)  # 这条边不会生成阵面
                                display_face(node0, node3)
                                AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)

                                # 第3个三角形
                                cell_num = cell_num + 1  # 多一个cell
                                Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                                Left_Cell_Index_new[node1_node1_nearby[1] - 1] = cell_num  # 补充阵面的左单元
                                AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                                AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])

                        elif hhh == Pbest_num[0]:
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # 第一个三角形
                            node_num_display(Node_x_new, Node_y_new)
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                            add_face(node1, Pbest_num[0], -1, cell_num)
                            display_face(node1, Pbest_num[0])
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)
                            add_face(Pbest_num[0], node2, cell_num+1, cell_num)  # 这条边不会生成阵面
                            display_face(Pbest_num[0], node2)

                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元

                            add_face(hhh, node3, cell_num+1, cell_num)
                            display_face(hhh, node3)

                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                            Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                            AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])


                        elif Pbest_num[0] <=len(Node_x_new)-1 : #判断最优点是不是在阵面上，如果在，直接连接，如果不在，则直接走到else中去,
                            #之所以减去1，是因为不能直接选择新点自己，而新点选择自己就是最后一种最常见的情况了
                            Node_x_new.pop()
                            Node_y_new.pop()
                            # 第一个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元

                            add_face(node1, Pbest_num[0], -1, cell_num)
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)
                            display_face(node1, Pbest_num[0])

                            cell_num = cell_num + 1  # 多一个cell
                            add_face(Pbest_num[0], node2, cell_num, cell_num - 1)  # 这条边不会生成阵面
                            display_face(Pbest_num[0], node2)

                            add_face(Pbest_num[0], node3, -1, cell_num)
                            display_face(Pbest_num[0], node3)
                            add_AFT_stack(Pbest_num[0], node3, face_num, cell_num)
                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                        else:
                            # 第一个三角形
                            cell_num = cell_num + 1  # 多一个cell
                            Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元

                            add_face(node1, Pbest_num[0], -1, cell_num)
                            add_AFT_stack(node1, Pbest_num[0], face_num, cell_num)
                            display_face(node1, Pbest_num[0])

                            cell_num = cell_num + 1  # 多一个cell
                            add_face(Pbest_num[0], node2, cell_num, cell_num - 1)  # 这条边不会生成阵面
                            display_face(Pbest_num[0], node2)

                            add_face(Pbest_num[0], node3, -1, cell_num)
                            display_face(Pbest_num[0], node3)
                            add_AFT_stack(Pbest_num[0], node3, face_num, cell_num)

                            Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                            AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)

                    time_m5_e = time.clock()
                    # print("模式5时间",time_m5_e-time_nn_s)

                elif model == 6:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    model_num6 = model_num6 + 1
                    #第一个三角形
                    if print_flag==1:
                        print("model=", model)
                    node2_node2_node2_nearby = node2_nearby_adv(node2_node2_nearby[0], node2_node2_nearby[1], ref_node2, ddd, fff)
                    hhh = node2_node2_node2_nearby[0]

                    cell_num = cell_num + 1  # 多一个cell
                    Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                    Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元

                    add_face(node0, node2, cell_num + 1, cell_num)
                    display_face(node0, node2)
                    AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)

                    if node0 !=hhh:

                        # 第二个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                        add_face(node0, node3, -1, cell_num)  #这条边不会生成阵面
                        add_AFT_stack(node0, node3, face_num, cell_num)
                        display_face(node0, node3)
                        AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                    else:
                        # 第二个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                        add_face(node0, node3, cell_num+1, cell_num)  #这条边不会生成阵面
                        display_face(node0, node3)

                        # 第3个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                        Left_Cell_Index_new[node2_node2_node2_nearby[1] - 1] = cell_num  # 补充阵面的左单元

                        AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, node2_node2_node2_nearby[1])

                    time_m6_e = time.clock()
                    # print("模式6时间",time_m6_e-time_nn_s)
                elif model == 7:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    #第一个三角形
                    model_num7 = model_num7 + 1
                    if print_flag==1:
                        print("model=", model)
                    if eee!=node4:
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元

                        add_face(node1, node4, -1, cell_num)
                        display_face(node1, node4)
                        add_AFT_stack(node1, node4, face_num, cell_num)

                        add_face(node4, node2, cell_num+1, cell_num)
                        display_face(node4, node2)

                        # 第二个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                        Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                        AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                    else:
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元

                        add_face(node1, node4, cell_num+2, cell_num)
                        display_face(node1, node4)
                        add_face(node4, node2, cell_num+1, cell_num)
                        display_face(node4, node2)

                        # 第二个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                        Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元


                        # 第二个三角形
                        cell_num = cell_num + 1  # 多一个cell
                        Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                        Left_Cell_Index_new[node1_node1_nearby[1]- 1] = cell_num  # 补充阵面的左单元

                        AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, node1_node1_nearby[1])
                        AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                        AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                    time_m7_e = time.clock()
                    # print("模式7时间",time_m7_e-time_nn_s)


                elif model == 8:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    #第一个三角形
                    model_num8 = model_num8 + 1
                    if print_flag==1:
                        print("model=", model)
                    cell_num = cell_num + 1  # 多一个cell
                    Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                    Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                    AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)

                    add_face(node0, node2, cell_num+1, cell_num)
                    display_face(node0, node2)

                    # 第二个三角形
                    cell_num = cell_num + 1  # 多一个cell
                    Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                    Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元

                    AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                    AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)
                    time_m8_e = time.clock()
                    # print("模式8时间",time_m8_e-time_nn_s)
                elif model == 9:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    #第一个三角形
                    model_num9 = model_num9 + 1
                    if print_flag==1:
                        print("model=", model)
                    cell_num = cell_num + 1  # 多一个cell
                    Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                    Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                    AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)

                    add_face(node1, node3, -1, cell_num)
                    display_face(node1, node3)
                    add_AFT_stack(node1, node3, face_num, cell_num)
                    time_m9_e = time.clock()
                    # print("模式9时间",time_m9_e-time_nn_s)
                elif model == 10:  # 1,可能会生成新点，将阵面第1\2点与新点连接，生成2阵面，更改数据结构
                    model_num10 = model_num10 + 1
                    #第一个三角形

                    if print_flag==1:
                        print("model=", model)
                    cell_num = cell_num + 1  # 多一个cell
                    Left_Cell_Index_new[AFT_stack[0][5] - 1] = cell_num  # 补充阵面的左单元
                    Left_Cell_Index_new[third_face_num - 1] = cell_num  # 补充阵面的左单元
                    AFT_stack = del_adv_elemnt(AFT_stack, third_face_num)
                    AFT_stack = del_adv_elemnt(AFT_stack, first_face_num)
                    AFT_stack = del_adv_elemnt(AFT_stack, four_face_num)

                    add_face(node1, node3, cell_num+1, cell_num)
                    display_face(node1, node3)

                    # 第二个三角形
                    cell_num = cell_num + 1  # 多一个cell
                    Left_Cell_Index_new[first_face_num - 1] = cell_num  # 补充阵面的左单元
                    Left_Cell_Index_new[four_face_num - 1] = cell_num  # 补充阵面的左单元
                    # print("时间10",time.clock()-time_s)
                    time_m10_e = time.clock()
                    # print("模式10时间",time_m10_e-time_nn_s)
        else:
            print('No checkpoint file found')


k=' '
# 参考 https://blog.csdn.net/u010305706/article/details/47837861

jg=open(r'./TEXT_20210516.txt','w+')
jg.truncate()   #清空文件
jg.write('Zone 1  Number of Nodes: '+str(len(Node_x_new))+'\n')
for i in  range(len(Node_x_new)):
    jg.write(str(Node_x_new[i])+k+str(Node_y_new[i])+'\n')

jg.write('Zone 2  Number of Face: '+str(len(Left_Cell_Index_new))+'\n')
for i in  range(len(Left_Cell_Index_new)):
    if i<Number_of_Boundary_Faces:
        jg.write(str(Left_Cell_Index_new[i]) + k + str(Right_Cell_Index_new[i]) + k + str(
            Face_Node_Index_new[2 * i+1]) + k + str(Face_Node_Index_new[2 * i]) + '\n')
    else:

        jg.write(str(Left_Cell_Index_new[i])+k+str(Right_Cell_Index_new[i])+k+str(
            Face_Node_Index_new[2*i])+k+str(Face_Node_Index_new[2*i+1])+'\n')



jg.close()
time_end=time.time()
print('time cost',time_end-time_start,'s')
print ("THE END")
