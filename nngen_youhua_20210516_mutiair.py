# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
# import nngen_adv_app_20191031

import random
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
import pandas as pd
import numpy as np
import math

# from pylab import *   #这句是为输出x和y的lim准备的，这里注销了，因为调试的时候会出现很多变量
#/home/lupeng/Downloads/anntest1/TEXT_20191112.txt# with open(r'/home/lupeng/Downloads/anntest1/nn_gen/TEXT_20191112.txt', 'r') as data_file:
with open(r'./TEXT_20210516.txt', 'r') as data_file:
    lines = data_file.readlines()
    #the fist data is invalid,so all copies  are 99 to distinguish them from the nomal data

################################################3寻找第二次含有含有"(13"，这是从line中搜索是一个字符字符所在的行#########
node_ln = []
face_ln=[]
nn = 0
K=1
nei=1
if nei==1:
    bandray=451
else:
    bandray=60
# jj=nngen_adv_app_20191031.Number_of_Boundary_Faces
for line1 in lines:   #把lines中的每一行提出来，放在line1中
    nn = nn + 1   #记录行数
    zi = "Nodes:"    #记录行数需要查找的字符串
    zii="Face:"
    elements = line1.split()  #把每一行打散
    if zi in elements:     #打散的字符串中查找是否有需要的字符串
        node_ln.append(nn)  #如果有把，其相应的行数记录下来
        print (node_ln)
    if zii in elements:     #打散的字符串中查找是否有需要的字符串
        face_ln.append(nn)  #如果有把，其相应的行数记录下来
        print (face_ln)
c=11
C=node_ln[0]
d=face_ln[0]
Number_of_Faces = int(lines[face_ln[0] - K].split()[5][:])
# get the string of first ten row 第六个空格后的数据 ,but cancel the last 2 character,at last string turn into int
Number_of_Nodes = int(lines[node_ln[0] - K].split()[5][:])
uu=1
Face_Node_Number=[]
Left_Cell_Index=[]
Right_Cell_Index=[]
Face_Node_Index=[]
Node_x=[]
Node_y=[]
cell_face=[]# 构建二维数组,
node_face=[]# 构建二维数组,
M=1
temp_Cell = []
Cell_Node_Number=[]
Cell_Node_Index=[]
node_2_face=[]
#1c11111######################Node_x,Node_y##################################################################################3

for i in range(Number_of_Nodes):                       #读取文件第22行开始的25个点的数据
    data = lines[i+node_ln[0]].split()[0]     #读取文件第22行的第0个空格后数据
    Node_x.append(float(data))                 #将data字符串形式转化为浮点型,点从x【0】开始
    data1 = lines[i+node_ln[0]].split()[1]     #读取文件第22行的空格第一个数据
    Node_y.append(float(data1))



for i in range(Number_of_Faces):  #16 boundary Face
    Face_Node_Number.append(2)

    data2 = lines[i + face_ln[0]].split()[2]  # 读取
    Face_Node_Index.append(int(data2, 10))  #后面的10表示10进制，如果读来的数据是16进制，则要写16


    data3 = lines[i + face_ln[0]].split()[3]  # face_ln[2]表示（13字符串出现第三次所在的行
    Face_Node_Index.append(int(data3, 10))

    data4 = lines[i + face_ln[0]].split()[0]  # 读取
    Left_Cell_Index.append(int(data4, 10))

    data5 = lines[i + face_ln[0]].split()[1]  # 读取
    Right_Cell_Index.append(int(data5, 10))
aaa=max(Left_Cell_Index)
aaa1=max(Right_Cell_Index)
Number_of_Cells = max([aaa, aaa1])
def dispaly():
    ########################以下显示边界点信息####################################
    # xlim(-90, 90)
    # # 设置y轴范围
    # ylim(-60, 60)
    for i in range(Number_of_Faces):  # 16 boundary Face
        jkl = Face_Node_Index[2 * i]  # 存的一条边起点序号
        hhj = Face_Node_Index[2 * i + 1]  # 存的一条边终点序号
        gx = [Node_x[jkl - 1], Node_x[hhj - 1]]  # 存的理论点号与实际存的点号相差1
        gy = [Node_y[jkl - 1], Node_y[hhj - 1]]  # 存的理论点号与实际存的点号相差1

        plt.plot(gx, gy, 'g-s', color='g', markerfacecolor='g', marker='')

    # x = np.array(Node_x)
    # y = np.array(Node_y)
    # i = 0
    for a, b in zip(x, y):
        print i
        i = i + 1
        plt.annotate('%s' % (i), xy=(a, b), color='k', xytext=(0, 0), textcoords='offset points')
    plt.axis("equal")
    #     # plt.show()



def cell_2_face():

    for i in  range(Number_of_Cells+1):#从0到1104，一共要1105
        cell_face.append([])
        for j in range(Number_of_Faces):
            if Left_Cell_Index[j]==i:
                cell_face[i].append(j) #j是序号，也就是边的编号
            if Right_Cell_Index[j] == i:
                cell_face[i].append(j)
temp_Cell = []
def cell_2_node():
    # 33333333######################Cell_Node_Index###################################
    for i in range(Number_of_Cells):  # 16 Cell
        global temp_Cell

        # print Number_of_Cells,i
        for j in range(Number_of_Faces):
            if Left_Cell_Index[j] == i + M:  # 以cell 1为例，查看哪些face以cell 1 为左部cell
                temp_Cell.append(Face_Node_Index[2 * j])  # 找到了，就将这条边的点号存储下来
                temp_Cell.append(Face_Node_Index[2 * j + 1])

        for j in range(Number_of_Faces):  # 原理同上，只是要反向而已
            if Right_Cell_Index[j] == i + M:
                temp_Cell.append(Face_Node_Index[2 * j + 1])
                temp_Cell.append(Face_Node_Index[2 * j])
        Cell_Node_Index.append([])
        Cell_Node_Index[i].append(list(set(temp_Cell)))
         # cell node index里面 就是存放的每个cell里面由那些点组成！
        temp_Cell = []

def quality():
    nine2one=0
    eight2nine=0
    seven2eight=0
    six2seven=0
    less2six=0
    for i in range(Number_of_Cells):
        if i==375:
            opop=1
        t1=Cell_Node_Index[i][0][0]
        t2=Cell_Node_Index[i][0][1]
        t3 = Cell_Node_Index[i][0][2]
        Q=2*Triangle_Quality_Judgment(Node_x[t1-1], Node_y[t1-1], Node_x[t2-1], Node_y[t2-1], Node_x[t3-1], Node_y[t3-1])
        print (i)
        if Q>=0.9 and Q<=1:
            nine2one=nine2one+1
        elif Q >= 0.8 and Q < 0.9:
            eight2nine = eight2nine + 1
        elif Q>=0.7 and Q<0.8:
            seven2eight=seven2eight+1
        elif Q>=0.6 and Q<0.7:
            six2seven=six2seven+1
        elif Q>=0.0 and Q<0.6:
            less2six=less2six+1
        else:
            print ("error Q <0")
    print ("Number_of_Cells:",Number_of_Cells)
    print ("nine2one:",nine2one)
    print ("eight2nine:",eight2nine)
    print ("seven2eight:",seven2eight)
    print ("six2seven:",six2seven)
    print ("less2six:",less2six)




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


def delaunay_test(x0,y0,x1,y1,x2,y2,x3,y3):
    #参考https://blog.csdn.net/Mr_Grit/article/details/46745357m   math.pow(12, 2)     # 求平方
    #参考https://blog.csdn.net/Mr_Grit/article/details/46745357m   math.pow(12, 2)     # 求平方
    m1=(math.pow(x0,2)+math.pow(y0,2))*y1+y0*(math.pow(x2,2)+math.pow(y2,2))+y2*(math.pow(x1,2)+math.pow(y1,2))-(math.pow(x2, 2) + math.pow(y2, 2)) * y1 - y0 * (math.pow(x1, 2) + math.pow(y1, 2)) - y2 * (math.pow(x0, 2) + math.pow(y0, 2))
    n1=2*((x0*y1+x2*y0+x1*y2)-(x2*y1+x1*y0+x0*y2))
    x=m1/n1
    m2=(math.pow(x1,2)+math.pow(y1,2))*x0+x2*(math.pow(x0,2)+math.pow(y0,2))+x1*(math.pow(x2,2)+math.pow(y2,2))- \
      (math.pow(x1, 2) + math.pow(y1, 2)) * x2 - x1 * (math.pow(x0, 2) + math.pow(y0, 2)) - x0 * (
                  math.pow(x2, 2) + math.pow(y2, 2))
    n2=2*((x0*y1+x2*y0+x1*y2)-(x2*y1+x1*y0+x0*y2))
    y=m2/n2


    r=math.sqrt(math.pow((x-x0),2)+math.pow((y-y0),2))#半径
    rr=math.sqrt(math.pow((x-x3),2)+math.pow((y-y3),2))#半径
    if r>=rr:
        return [0,x,y]  #表示外接圆半径大于圆心到点的距离，则点被包含在圆中。delaunay测试不通过
    else:
        return [1,x,y]  #delaunay测试通过


def node_2_face_fun():
    for i in range(Number_of_Nodes):
        node_2_face.append([])
        for j in range(Number_of_Faces):
            if Face_Node_Index[2*j]==i+1 or Face_Node_Index[2*j+1]==i+1: #与点号为IIi+1的点相连的face号记录下来 node_2_face[i]的2维数组中
                node_2_face[i].append(j+1)

def tanhuang():

    for l in range(50):#做弹簧变形50次
        # for i in range(bandray,Number_of_Nodes,1):  #思路是把除边界以为的点进行弹簧变形，本质上就是找到一个点的所有临点，求临点的平均值代替现在点的坐标xy
        for i in range(bandray, Number_of_Nodes, 1):
            node_2_face_temp=node_2_face[i]
            node_temp = []
            for j in range(len(node_2_face_temp)):
                xxx=Face_Node_Index[2*(node_2_face_temp[j]-1)]
                xxx1=Face_Node_Index[2*(node_2_face_temp[j]-1)+1]

                node_temp.append(xxx)
                node_temp.append(xxx1)

            xxx2=[i+1]

            node_nearby = list(set(node_temp).difference(set(xxx2)))
            x_temp=0
            y_temp=0
            for k in range(len(node_nearby)):
                x_temp=x_temp+Node_x[node_nearby[k]-1]
                y_temp = y_temp + Node_y[node_nearby[k] - 1]

            x_temp=x_temp/len(node_nearby)
            y_temp = y_temp / len(node_nearby)
            Node_x[i]=x_temp
            Node_y[i] = y_temp
    dispaly()
    plt.show()

def duijiaoxian_change():
    global flag
    for i in range(0, Number_of_Faces):
        if Right_Cell_Index[i] != 0:  # 边界边不做对角线判断，因其右边没有cell
            # 以下几行代码求某条边左右cell对应的点
            print("face_num", i + 1)  # 实际边号需要加1
            if test_flag == 1:
                node1 = Face_Node_Index[2 * i]
                node2 = Face_Node_Index[2 * i + 1]

            ccc1 = Left_Cell_Index[i]  # 某条边的左cell
            ddd1 = Right_Cell_Index[i]  # 某条边的右cell
            ccc = Cell_Node_Index[ccc1 - 1]  # 左右cell对应的点
            ddd = Cell_Node_Index[ddd1 - 1]
            ccc = [ccc[0][0], ccc[0][1], ccc[0][2]]  # 左右cell对应的点，转化为一维情况
            ddd = [ddd[0][0], ddd[0][1], ddd[0][2]]
            # face所对应的点，以及face左右的对角点
            face_node = [Face_Node_Index[2 * i], Face_Node_Index[2 * i + 1]]
            dingdian1 = list(set(ccc).difference(set(face_node)))
            dingdian2 = list(set(ddd).difference(set(face_node)))
            print("node1, node2,dingdian1,dingdian2", node1, node2, dingdian1[0], dingdian2[0])
            Lcn = ccc  # left cell node simlify

            test1 = delaunay_test(Node_x[Lcn[0] - 1], Node_y[Lcn[0] - 1], Node_x[Lcn[1] - 1], Node_y[Lcn[1] - 1],
                                  Node_x[Lcn[2] - 1], Node_y[Lcn[2] - 1], \
                                  Node_x[dingdian2[0] - 1], Node_y[dingdian2[0] - 1])
            if test1[0] == 0:  # 需要变换face的数据结构（face的左右点，face的left cell 和right cell）
                #
                flag = flag + 1
                print('对角线变换次数： %i' % flag)
                print("其中一个单元的第一个点位置", Node_x[Lcn[0] - 1], Node_y[Lcn[0] - 1])
                Face_Node_Index[2 * i] = dingdian1[0]
                Face_Node_Index[2 * i + 1] = dingdian2[0]  # 将边i的左右两点换成对角线点，因为现在的不满足delaunay法则

                ccc1 = Left_Cell_Index[i]  # 某条边i的左cell
                ddd1 = Right_Cell_Index[i]  # 某条边i的右cell

                Cell_Node_Index[ccc1 - 1] = [[dingdian1[0], dingdian2[0], face_node[1]]]
                Cell_Node_Index[ddd1 - 1] = [[dingdian1[0], dingdian2[0], face_node[0]]]

                mmm1 = node_2_face[face_node[0] - 1]
                nnn1 = node_2_face[dingdian1[0] - 1]
                face_flag1 = list(set(mmm1).intersection(set(nnn1)))
                print("顶点1和以前边的一个点的边号", face_flag1)  # 找到顶点i和以前边的一个点的边号
                mmm2 = node_2_face[face_node[1] - 1]
                nnn2 = node_2_face[dingdian2[0] - 1]
                face_flag2 = list(set(mmm2).intersection(set(nnn2)))
                print("顶点2和以前边的一个点的边号", face_flag2)

                # node_2_face 数据更新
                bian_hao = set(mmm1).intersection(set(mmm2))
                node1_to_face = set(mmm1) - bian_hao
                node2_to_face = set(mmm2) - bian_hao
                node_2_face[face_node[0] - 1] = list(node1_to_face)
                node_2_face[face_node[1] - 1] = list(node2_to_face)

                dingdian1_to_face = set(nnn1) | bian_hao
                dingdian2_to_face = set(nnn2) | bian_hao
                node_2_face[dingdian1[0] - 1] = list(dingdian1_to_face)
                node_2_face[dingdian2[0] - 1] = list(dingdian2_to_face)

                if Left_Cell_Index[face_flag1[0] - 1] == ccc1:  # 某条边的左cell
                    Left_Cell_Index[face_flag1[0] - 1] = ddd1
                if Right_Cell_Index[face_flag1[0] - 1] == ccc1:  # 某条边的左cell
                    Right_Cell_Index[face_flag1[0] - 1] = ddd1

                if Left_Cell_Index[face_flag2[0] - 1] == ddd1:  # 某条边的左cell
                    Left_Cell_Index[face_flag2[0] - 1] = ccc1
                if Right_Cell_Index[face_flag2[0] - 1] == ddd1:  # 某条边的左cell
                    Right_Cell_Index[face_flag2[0] - 1] = ccc1



# dispaly()
# plt.show()

# cell_2_face()
# print ("mmmm")
cell_2_node()
cell_node_test=[]
for i in range(Number_of_Cells):
    if len(Cell_Node_Index[i][0])!=3:
        print ("cell num,error node",i,Cell_Node_Index[i][0])

print ("kkk")
quality()
node_2_face_fun()
face_node=[]
test_flag=1
print ("uuu")

flag=0


# dispaly()
# plt.show()
print ('flag is %i' %flag)
duijiaoxian_change()
print ('after transform')
tanhuang()
quality()
duijiaoxian_change()

quality()
dispaly()
plt.show()
print ("the end")