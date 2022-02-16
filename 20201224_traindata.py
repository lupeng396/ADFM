# -*- coding: utf-8 -*-
""""
(1 "Exported from Pointwise")
(0 "08:12:08  Tue Jun 25 2019")

(0 "Dimension : 2")
(2 2)

(0 "Number of Nodes : 564")
(10 (0 1 234 0 2))

(0 "Total Number of Faces : 1092")
(0 "       Boundary Faces : 100")
(0 "       Interior Faces : 992")
(13 (0 1 444 0))

(0 "Total Number of Cells : 529")
(0 "            Tri cells : 32")
(0 "           Quad cells : 497")
(12 (0 1 211 0))

(0 "Zone 1  Number of Nodes : 564")
编程思路，由pontwise生成一个5x5的标准结构化网格，得到其grid文件，改名为txt，然后读入每个点的信息，以及非边界和边界信息，最后由已知信息构建其他
的几何结构信息，如网格与点，网格与边的信息等。by lupeng 2019.6.16
"""
import matplotlib.pyplot as plt
import random
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

import pandas as pd
import numpy as np
import math

front_size=15
fig_flag=0
flag_1=0
flag_2=0
flag_3=0
flag_4=0
flag_5=0
flag_6=0
flag_7=0
flag_8=0
flag_9=0
flag_10=0
flag_error=0
display_flag=0
with open(r'../datainput/20210312_two_front.cas', 'r') as data_file:
# with open(r'/home/lupeng/Downloads/anntest1/datainput/circle.cas', 'r') as data_file:
    lines = data_file.readlines()
    #the fist data is invalid,so all copies  are  to distinguish them from the nomal data
################################################3寻找第二次含有含有"(13"，这是从line中搜索是一个字符字符所在的行#########
face_ln=[]
nn=0
for line1 in lines:
    nn=nn+1
    zi="(13"
    elements=line1.split()
    if zi in elements:
        face_ln.append(nn)
        print (face_ln)

###################################################
filelen=len(lines)
K = 1
M = 1
Dimension=lines[4-K].split()[3][:-2] #去掉最后的分号和括号
Dimension=int(Dimension)

Number_of_Faces=int(lines[10-K].split()[6][:-2])   #get the string of first ten row 第六个空格后的数据 ,but cancel the last 2 character,at last string turn into int
Number_of_Nodes=int(lines[7-K].split()[5][:-2])

Number_of_Boundary_Faces=lines[11-K].split()[5]#取第八行第四个空格后的字符串，取出最后两位，然后强制转化为int
Number_of_Boundary_Faces=int(Number_of_Boundary_Faces[:-2])

Number_of_Interior_Faces=lines[12-K].split()[5]#
Number_of_Interior_Faces=int(Number_of_Interior_Faces[:-2])

Number_of_Cells=lines[15-K].split()[6]#
Number_of_Cells=int(Number_of_Cells[:-2])

Node_x=[]        #点的x坐标
Node_y=[]

p2p=[]
delay = 1
Face_Node_Number=[]
Face_Node_Index=[]

Left_Cell_Index=[]
Right_Cell_Index=[]

Cell_Node_Number=[]
Cell_Node_Index=[]

temp_Cell=[]
temp_node_face=[]
count=0
columns = 4   #lie
Node_Face=[]
Cun_nearbynode_nearbynode=[]
Cun_nearbynode_nearbynode_1=[]
train_data_output=[]
traindata_out=[]
# Node_Face=[]
m1=[]
n1=[]
#111111######################Node_x,Node_y##################################################################################3
for i in range(Number_of_Nodes):                       #读取文件第22行开始的25个点的数据
    data = round(float(lines[i+21].split()[0]),4)     #读取文件第22行的第0个空格后数据
    Node_x.append(data)                 #将data字符串形式转化为浮点型,点从x【0】开始
    data1 = round(float(lines[i+21].split()[1]),4)     #读取文件第22行的空格第一个数据
    Node_y.append(data1)
#################################################################################################################################
#2222222###################Face_Node_Number  Face_Node_Index  Left_Cell_Index  Right_Cell_Index#################################
for i in range(Number_of_Boundary_Faces):  #16 boundary Face
    Face_Node_Number.append(2)

    data2 = lines[i + face_ln[2]].split()[0]  # face_ln[2]表示（13字符串出现第三次所在的行
    Face_Node_Index.append(int(data2, 16))

    data3 = lines[i + face_ln[2]].split()[1]  # 读取
    Face_Node_Index.append(int(data3, 16))

    data4 = lines[i + face_ln[2]].split()[2]  # 读取
    Left_Cell_Index.append(int(data4, 16))

    data5 = lines[i + face_ln[2]].split()[3]  # 读取
    Right_Cell_Index.append(int(data5, 16))


# 10进制转16进制: hex(16) == > 0x10
# 16进制转10进制: int('0x10', 16) == > 16
#以下读入非边界边的信息
for i in range(Number_of_Interior_Faces):  #24 Interior Face
    Face_Node_Number.append(2)

    data2 = lines[i + face_ln[1]].split()[0]  # 读取 face_ln[1]表示（13字符串出现第二次所在的行
    Face_Node_Index.append(int(data2, 16))
    data3 = lines[i + face_ln[1]].split()[1]  # 读取文件
    Face_Node_Index.append(int(data3, 16))

    data4 = lines[i + face_ln[1]].split()[2]  # 读取
    Left_Cell_Index.append(int(data4, 16))

    data5 = lines[i + face_ln[1]].split()[3]  # 读取
    Right_Cell_Index.append(int(data5, 16))


#33333333######################Cell_Node_Index###################################
for i in range(Number_of_Cells):  #16 Cell
    for j in range(Number_of_Interior_Faces):
        if  Left_Cell_Index[Number_of_Boundary_Faces+j]==i+M:#以cell 1为例，查看哪些face以cell 1 为左部cell
            temp_Cell.append(Face_Node_Index[2*(Number_of_Boundary_Faces+j)])  #找到了，就将这条边的点号存储下来
            temp_Cell.append(Face_Node_Index[2*(Number_of_Boundary_Faces+j)+1])

    for j in range(Number_of_Interior_Faces):#原理同上，只是要反向而已
        if Right_Cell_Index[Number_of_Boundary_Faces+j] == i+M:
            temp_Cell.append(Face_Node_Index[2*(Number_of_Boundary_Faces+j)+1])
            temp_Cell.append(Face_Node_Index[2*(Number_of_Boundary_Faces+j)])

    for j in range(Number_of_Boundary_Faces):
        if  Left_Cell_Index[j] == i+M:
            temp_Cell.append(Face_Node_Index[2*j])
            temp_Cell.append(Face_Node_Index[2*j+1])
    if (len(temp_Cell)== 8):

        if temp_Cell[5]!=temp_Cell[6]:
            temp_cp6 = temp_Cell[6]
            temp_cp7 = temp_Cell[7]
            temp_Cell[6] = temp_Cell[4]
            temp_Cell[7] = temp_Cell[5]
            temp_Cell[4] = temp_cp6
            temp_Cell[5] = temp_cp7

        Cell_Node_Number.append(4)
    #    Cell_Node_Index.append([])

        for column in range(0,8,2):
            num = temp_Cell[column]
            Cell_Node_Index.append(num)
        temp_Cell = []
    else:
        if temp_Cell[3] != temp_Cell[4]:
            temp_cp6 = temp_Cell[4]
            temp_cp7 = temp_Cell[5]
            temp_Cell[4] = temp_Cell[2]
            temp_Cell[5] = temp_Cell[3]
            temp_Cell[2] = temp_cp6
            temp_Cell[3] = temp_cp7

        Cell_Node_Number.append(3)
        #    Cell_Node_Index.append([])

        for column in range(0, 6, 2):
            num = temp_Cell[column]
            Cell_Node_Index.append(num) #cell node index里面 就是存放的每个cell里面由那些点组成！
        temp_Cell = []

testp1=0
testp1=10
##############################Node_Face[]################################################################################
#以下代码功能是保存了所有点到边的信息，不过信息为
#建立每一个点的周围的有哪些边，最后返回值跟number有关
def Neighbourhood_Node(number):
    global temp_node_face
    global count
    global bianhao
    Node_Face=[]
    bianhao=[]
    for i in range(Number_of_Nodes):
        for j in range(Number_of_Boundary_Faces): #查看所有外部边的左边点或者右边点是否有等于
            if  (Face_Node_Index[2*j]==i+M) or(Face_Node_Index[2*j+1]==i+M):
                temp_node_face.append(j+1)
                count = count+1
        for j in range(Number_of_Interior_Faces):
            if (Face_Node_Index[2*(Number_of_Boundary_Faces+j)] == i + M) or (Face_Node_Index[2*(Number_of_Boundary_Faces+j)+1] == i + M):
                temp_node_face.append(Number_of_Boundary_Faces+j+1)
                count = count + 1

        Node_Face.append([])
        for k in range(count):
            num = temp_node_face[k]
            Node_Face[i].append(num)  #二维数组存储
        temp_node_face = []
        count=0
    bianhao=Node_Face[number-1]
    return bianhao

def Node_Nearby_Node(face_index_number,input): #把每一条边找出来，找到他们的两个点，去除当前点，就是当前点所连接的边
    temp2=[]
    p2p=[]
    for j in range(face_index_number): #每条边里面有哪些点
        p2p.append(Face_Node_Index[2*(pp[j]-1)])
        p2p.append(Face_Node_Index[2*(pp[j]-1)+1])
    p2p_len=len(p2p)
    for j in range(p2p_len):
        ceshi=p2p[j]
        if input!=ceshi:
            temp2.append(p2p[j])
    return temp2
def Node_zuobiao(number):
    temp3=[]
    for j in range(number):
        temp3.append(Node_x[temp12[j]-1])
        temp3.append(Node_y[temp12[j]-1])
    return temp3
####################################################################################################################3
#整体平移函数
def Translation(x,y,x0,y0):#x,y表示要任何要平移的点，x0，y0表示将以何点为圆心作为参考点
    x_new=x-x0
    y_new=y-y0
    return [x_new,y_new]

def Anti_Translation(x,y,x0,y0):#x,y表示要任何要anti平移的点，x0，y0表示将以何点为圆心作为参考点
    x_new=x+x0
    y_new=y+y0
    return [x_new,y_new]
#缩放函数，归一化处理x_start,y_start,x_end,y_end,x,y分别表示坐标系x轴起点与终点坐标值，xy为任意需要变换的点
def Scaling(x_start,y_start,x_end,y_end,x,y):  #注意xy是经过平移的点
    len=math.sqrt(((x_end-x_start)**2)+((y_end-y_start)**2))
    x_new = (1/len) * x
    y_new = (1 / len) * y
    return [x_new,y_new]

def Anti_Scaling(x_start,y_start,x_end,y_end,x,y):  #注意xy是经过反旋转的的点
    len=math.sqrt(((x_end-x_start)**2)+((y_end-y_start)**2))
    x_new = (len) * x
    y_new = (len) * y
    return [x_new,y_new]


def Rotation(x_end,y_end,x_rot,y_rot):#x_end,y_end为第一步平移后face另外一个点的值以便求与全局坐标系求夹角ceta，
    # xrot yrot为任意需要旋转的坐标

    # 相当于勾股定理，求得斜线的长度
    x = np.array([x_end,y_end])  # 向量1  x'
    y = np.array([1, 0])  # 全局坐标系x
    jj = x.dot(x)  # 向量点乘

    Lx = np.sqrt(x.dot(x))  # 向量点乘
    Ly = np.sqrt(y.dot(y))  # 向量点乘
    kk = x.dot(y)  # 向量点乘
    cos_angle = x.dot(y) / (Lx * Ly)
    sin_angle = y_end / (Lx * Ly)
    # 说明https://zhidao.baidu.com/question/1705964907383367940.html
    if cos_angle > 0 and sin_angle > 0:  # 1象限
        seta2 = math.asin(sin_angle)
        seta2 = 360 * seta2 / (2 * math.pi)
    if cos_angle < 0 and sin_angle > 0:  # 2象限
        seta2 = math.asin(sin_angle)
        seta2 = 180 - 360 * seta2 / (2 * math.pi)
    if cos_angle < 0 and sin_angle < 0:  # 3象限
        seta2 = math.asin(sin_angle)
        seta2 = 180 - 360 * seta2 / (2 * math.pi)
    if cos_angle > 0 and sin_angle < 0:  # 4象限
        seta2 = math.asin(sin_angle)
        seta2 = 360 + 360 * seta2 / (2 * math.pi)

    x_new = round(cos_angle * x_rot + sin_angle * y_rot, 2)
    y_new = round(-sin_angle * x_rot + cos_angle * y_rot, 2)
    return [x_new, y_new]

def Anti_Rotation(x_end,y_end,x_rot,y_rot):#x_end,y_end为第一步平移后face另外一个点的值以便求与全局坐标系求夹角ceta，x_rot,y_rot是需要反旋转的量
    # xrot yrot为任意需要旋转的坐标

    # 相当于勾股定理，求得斜线的长度
    x = np.array([x_end,y_end])  # 向量1  x'
    y = np.array([1, 0])  # 全局坐标系x
    jj = x.dot(x)  # 向量点乘

    Lx = np.sqrt(x.dot(x))  # 向量点乘
    Ly = np.sqrt(y.dot(y))  # 向量点乘
    kk = x.dot(y)  # 向量点乘
    cos_angle = x.dot(y) / (Lx * Ly)
    sin_angle = y_end / (Lx * Ly)
    # 说明https://zhidao.baidu.com/question/1705964907383367940.html
    if cos_angle > 0 and sin_angle > 0:  # 1象限
        seta2 = math.asin(sin_angle)
        seta2 = 360 * seta2 / (2 * math.pi)
    if cos_angle < 0 and sin_angle > 0:  # 2象限
        seta2 = math.asin(sin_angle)
        seta2 = 180 - 360 * seta2 / (2 * math.pi)
    if cos_angle < 0 and sin_angle < 0:  # 3象限
        seta2 = math.asin(sin_angle)
        seta2 = 180 - 360 * seta2 / (2 * math.pi)
    if cos_angle > 0 and sin_angle < 0:  # 4象限
        seta2 = math.asin(sin_angle)
        seta2 = 360 + 360 * seta2 / (2 * math.pi)
    x_new = round(cos_angle * x_rot - sin_angle * y_rot, 2)
    y_new = round(sin_angle * x_rot + cos_angle * y_rot, 2)

    return [x_new, y_new]

def len_calcu(x_start,y_start,x_end,y_end):
    len = math.sqrt(((x_end - x_start) ** 2) + ((y_end - y_start) ** 2))
    return len



def dispaly():
    # plt.figure(figsize=(6, 6))
    global fig_flag

    ########################以下显示边界点信息####################################
    for i in range(Number_of_Boundary_Faces):  # 16 boundary Face
        jkl = Face_Node_Index[2 * i]  # 存的一条边起点序号
        hhj = Face_Node_Index[2 * i + 1]  # 存的一条边终点序号
        gx = [Node_x[Face_Node_Index[2 * i] - 1], Node_x[Face_Node_Index[2 * i + 1] - 1]]  # 存的理论点号与实际存的点号相差1
        gy = [Node_y[Face_Node_Index[2 * i] - 1], Node_y[Face_Node_Index[2 * i + 1] - 1]]  # 存的理论点号与实际存的点号相差1

        plt.plot(gx, gy, 'g-s', color='g', markerfacecolor='g', marker='o')
    #########################以下显示内部边信息##################################
    for i in range(Number_of_Interior_Faces):  # 16 boundary Face
        jkl = Face_Node_Index[2 * (i + Number_of_Boundary_Faces)]  # 存的一条边起点序号
        hhj = Face_Node_Index[2 * (i + Number_of_Boundary_Faces) + 1]  # 存的一条边终点序号
        gix = [Node_x[Face_Node_Index[2 * (i + Number_of_Boundary_Faces)] - 1],
               Node_x[Face_Node_Index[2 * (i + Number_of_Boundary_Faces) + 1] - 1]]
        giy = [Node_y[Face_Node_Index[2 * (i + Number_of_Boundary_Faces)] - 1],
               Node_y[Face_Node_Index[2 * (i + Number_of_Boundary_Faces) + 1] - 1]]

        plt.plot(gix, giy, 'g--', color='g', markerfacecolor='g', marker='*')
    #############################以下显示所有的点号##################################
    x = np.array(Node_x)
    y = np.array(Node_y)
    i = 0
    for a, b in zip(x, y):
        i = i + 1
        plt.annotate('%s' % (i), xy=(a, b),color='k', xytext=(0, 0),textcoords='offset points',fontsize=front_size)
        #https://blog.csdn.net/you_are_my_dream/article/details/53454549

    # plt.xlim((-1.2, 1.2))
    # plt.xticks(np.linspace(-11,11,23,endpoint=True))
    # plt.ylim((-1.2, 1.2))
    # plt.yticks(np.linspace(-11,11,23,endpoint=True))
    plt.axis("equal")


    fig_flag=fig_flag+1
    plt.savefig(r"./train_fig/Chart {}.png".format(fig_flag),dpi=80) #保存分辨率为80的图片
    # plt.show()
    return
#############################以上显示所有的点号##################################
def front_face2_in_boundary(point1,point2):
    in_boundary_flag = 0
    for i in range(Number_of_Boundary_Faces):
        if (Face_Node_Index[2 * i]==point1 and Face_Node_Index[2 * i+1]==point2) or(Face_Node_Index[2 * i]==point2 and Face_Node_Index[2 * i+1]==point1):
            in_boundary_flag=1
            break
    return in_boundary_flag

def duijiao_of_face(point1,point2):
    a=1
    for i in range(Number_of_Interior_Faces):  # 16 boundary Face
        node1 = Face_Node_Index[2 * (i + Number_of_Boundary_Faces)]  # 存的一条边起点序号
        node2 = Face_Node_Index[2 * (i + Number_of_Boundary_Faces) + 1]  # 存的一条边终点序号
        if (node1==point1 and node2==point2) or (node1==point2 and node2==point1):
            inputface=i+ Number_of_Boundary_Faces+1
            train_point1 = Face_Node_Index[2 * (inputface-1)]  # 得到face——index里面存的第一个点号
            train_point2 = Face_Node_Index[2 * (inputface-1) + 1]  # 得到face——index里面存的第二个点号
            face_Node12 = [train_point1, train_point2]
            # train_point1_x = Node_x[train_point1 - 1]  # 得到以上两点的坐标值
            # train_point1_y = Node_y[train_point1 - 1]
            # train_point2_x = Node_x[train_point2 - 1]
            # train_point2_y = Node_y[train_point2 - 1]
            train_point1_x = Node_x[point1 - 1]  # 双阵面的第二个阵面的情况
            train_point1_y = Node_y[point1 - 1]
            train_point2_x = Node_x[point2 - 1]
            train_point2_y = Node_y[point2 - 1]
            # 阵面显示用的坐标
            advance_display_x = [train_point1_x, train_point2_x]
            advance_display_y = [train_point1_y, train_point2_y]
            # 测试旋转平移缩放是不是正确的
            train_point2_tans = Translation(train_point2_x, train_point2_y, train_point1_x, train_point1_y)
            train_point2_Scaling = Scaling(train_point1_x, train_point1_y,
                                           train_point2_x, train_point2_y, train_point2_tans[0],
                                           train_point2_tans[1])  # 应该是0 0
            train_point2_Rotation = Rotation(train_point2_Scaling[0], train_point2_Scaling[1], train_point2_Scaling[0],
                                             train_point2_Scaling[1])
            # 找face的左右单元的对角点
            left_cell_facein = Left_Cell_Index[inputface - 1]
            left_cell_facein_3node = [Cell_Node_Index[3 * (left_cell_facein - 1)],
                                      Cell_Node_Index[3 * (left_cell_facein - 1) + 1],
                                      Cell_Node_Index[3 * (left_cell_facein - 1) + 2]]

            # 为显示左单元提供数据,由于plt plt.fill_between(x, y,k, facecolor="b")中间函数的限制，在理解他基础上，需要变换2019.7.9
            left_cell_facein_3node_x = [Node_x[left_cell_facein_3node[0] - 1], Node_x[left_cell_facein_3node[1] - 1],
                                        Node_x[left_cell_facein_3node[2] - 1]]

            left_cell_facein_3node_y = [Node_y[left_cell_facein_3node[0] - 1], Node_y[left_cell_facein_3node[1] - 1],
                                        Node_y[left_cell_facein_3node[2] - 1]]

            x1 = Node_x[left_cell_facein_3node[0] - 1]
            y1 = Node_y[left_cell_facein_3node[0] - 1]
            x2 = Node_x[left_cell_facein_3node[2] - 1]
            y2 = Node_y[left_cell_facein_3node[2] - 1]
            x3 = Node_x[left_cell_facein_3node[1] - 1]
            if x2 - x1==0:
                x2=x1+0.0001
            b = (y2 * (x3 - x1) + y1 * (x2 - x3)) / (x2 - x1)
            left_cell_facein_3node_y2 = [Node_y[left_cell_facein_3node[0] - 1], b,
                                         Node_y[left_cell_facein_3node[2] - 1]]

            # #################################################################################################
            left_cell_facein_duijiao = list(set(left_cell_facein_3node) - set(face_Node12))
            left_cell_facein_duijiao_x = Node_x[left_cell_facein_duijiao[0] - 1]
            left_cell_facein_duijiao_y = Node_y[left_cell_facein_duijiao[0] - 1]
            left_cell_duijiao_tans = Translation(left_cell_facein_duijiao_x, left_cell_facein_duijiao_y, train_point1_x,
                                                 train_point1_y)
            left_cell_duijiao_Scaling = Scaling(train_point1_x, train_point1_y, train_point2_x,
                                                train_point2_y, left_cell_duijiao_tans[0],
                                                left_cell_duijiao_tans[1])  # 应该是0 0
            left_cell_duijiao_Rotation = Rotation(train_point2_Scaling[0], train_point2_Scaling[1],
                                                  left_cell_duijiao_Scaling[0], left_cell_duijiao_Scaling[1])
            right_cell_facein = Right_Cell_Index[inputface - 1]

            if right_cell_facein != 0:
                right_cell_facein_3node = [Cell_Node_Index[3 * (right_cell_facein - 1)],
                                           Cell_Node_Index[3 * (right_cell_facein - 1) + 1],
                                           Cell_Node_Index[3 * (right_cell_facein - 1) + 2]]
                right_cell_facein_duijiao = list(set(right_cell_facein_3node) - set(face_Node12))
                right_cell_facein_duijiao_x = Node_x[right_cell_facein_duijiao[0] - 1]
                right_cell_facein_duijiao_y = Node_y[right_cell_facein_duijiao[0] - 1]
                # 将以上两个对角点在以阵面为基准坐标中，找出在基准坐标上方的点，因为这样的才，，满足右手定则

                right_cell_duijiao_tans = Translation(right_cell_facein_duijiao_x, right_cell_facein_duijiao_y,
                                                      train_point1_x, train_point1_y)
                right_cell_duijiao_Scaling = Scaling(train_point1_x, train_point1_y,
                                                     train_point2_x, train_point2_y, right_cell_duijiao_tans[0],
                                                     right_cell_duijiao_tans[1])  # 应该是0 0
                right_cell_duijiao_Rotation = Rotation(train_point2_Scaling[0],
                                                       train_point2_Scaling[1], right_cell_duijiao_Scaling[0],
                                                       right_cell_duijiao_Scaling[1])

            if left_cell_duijiao_Rotation[1] > 0:  # 将以上两个对角点在以阵面为基准坐标中，找出在基准坐标上方的点，因为这样的才，，满足右手定则
                new_point = left_cell_duijiao_Rotation
                new_point_old_zuobiao = left_cell_facein_duijiao[0]
            if right_cell_duijiao_Rotation[1]>0:
                new_point=right_cell_duijiao_Rotation
                new_point_old_zuobiao = right_cell_facein_duijiao[0]
    return new_point_old_zuobiao
def display_tri_feature(left_cell_facein_3node):
    #为显示左单元提供数据,由于plt plt.fill_between(x, y,k, facecolor="b")中间函数的限制，在理解他基础上，需要变换2019.7.9
    left_cell_facein_3node_x=[Node_x[left_cell_facein_3node[0]-1],Node_x[left_cell_facein_3node[1]-1],
                              Node_x[left_cell_facein_3node[2]-1]]


    left_cell_facein_3node_y = [Node_y[left_cell_facein_3node[0]-1],Node_y[left_cell_facein_3node[1]-1],
                              Node_y[left_cell_facein_3node[2]-1]]

    x1=Node_x[left_cell_facein_3node[0]-1]
    y1=Node_y[left_cell_facein_3node[0]-1]
    x2=Node_x[left_cell_facein_3node[2]-1]
    y2=Node_y[left_cell_facein_3node[2]-1]
    x3=Node_x[left_cell_facein_3node[1]-1]
    if x2-x1==0:
        x2 =x1+0.00001

    b=(y2*(x3-x1)+y1*(x2-x3))/(x2-x1)
    left_cell_facein_3node_y2 = [Node_y[left_cell_facein_3node[0] - 1], b,
                                Node_y[left_cell_facein_3node[2] - 1]]
    return [left_cell_facein_3node_x,left_cell_facein_3node_y,left_cell_facein_3node_y2]

def aotu_judge_func(node11,node12,node21,node22):
    global Node_x,Node_y


    x_shiliang1 = (Node_x[node12 - 1] - Node_x[node11 - 1])
    y_shiliang1 = (Node_y[node12 - 1] - Node_y[node11 - 1])

    x_shiliang2 = (Node_x[node22 - 1] - Node_x[node21 - 1])
    y_shiliang2 = (Node_y[node22 - 1] - Node_y[node21 - 1])
     #https://www.cnblogs.com/qingsunny/archive/2013/08/11/3251179.html
    # 求两个向量的凹凸性，采用两个向量的叉积 由 ijk 与abc xyz形成的行列式的正负行可以判断逆时针与顺时针，从而可以判断凹凸m=ay-xb，因为c与z为0
    aotu = x_shiliang1 * y_shiliang2 - x_shiliang2 * y_shiliang1  # ＞0为凹，＜0为凸
    xxx = np.array([x_shiliang1, y_shiliang1])
    yyy = np.array([x_shiliang2, y_shiliang2])
    # 两个向量
    Lx = np.sqrt(xxx.dot(xxx))  # 两个向量长度
    Ly = np.sqrt(yyy.dot(yyy))

    cos_angle = xxx.dot(yyy) / (Lx * Ly)  # https://blog.csdn.net/qq_42423940/article/details/83757427
    if cos_angle>1:
        cos_angle=1
    if cos_angle<-1:
        cos_angle=-1
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi  # 方向向量的偏转角度
    return aotu



tiqu_node=[]
right_cell_node=[]
left_cell_node=[]
km1=0
km2=0
M3=0
node3_nearby=[]
show_flag=1
# dispaly()
if __name__=='__main__':
    # num=[5-1]   #测试用，
    ceshi=1
    if ceshi==0:
        num=1
    else:
        num=Number_of_Interior_Faces+Number_of_Boundary_Faces
    for JKL in range(num): #下面是实际采集数据遍历用
        print("bian num",JKL)
        # inputface=JKL
        inputface = JKL
        train_point1=Face_Node_Index[2*(inputface-1)] #得到face——index里面存的第一个点号
        train_point2=Face_Node_Index[2*(inputface-1)+1]#得到face——index里面存的第二个点号
        face_Node12=[train_point1,train_point2]
        train_point1_x=Node_x[train_point1-1]#得到以上两点的坐标值
        train_point1_y=Node_y[train_point1-1]
        train_point2_x=Node_x[train_point2-1]
        train_point2_y=Node_y[train_point2-1]
        #阵面1显示用的坐标
        advance_display_x=[train_point1_x,train_point2_x]
        advance_display_y = [train_point1_y, train_point2_y]
        # 测试旋转平移缩放是不是正确的
        train_point2_tans=Translation(train_point2_x,train_point2_y,train_point1_x,train_point1_y)
        train_point2_Scaling = Scaling(train_point1_x,train_point1_y,
                                       train_point2_x,train_point2_y, train_point2_tans[0], train_point2_tans[1])  # 应该是0 0
        train_point2_Rotation=Rotation(train_point2_Scaling[0],train_point2_Scaling[1],train_point2_Scaling[0],train_point2_Scaling[1])
                       # 找face的左右单元的对角点
        left_cell_facein=Left_Cell_Index[inputface-1]
        left_cell_facein_3node=[Cell_Node_Index[3*(left_cell_facein-1)],
                                Cell_Node_Index[3*(left_cell_facein-1)+1],Cell_Node_Index[3*(left_cell_facein-1)+2]]

        display_tri_feature1=display_tri_feature(left_cell_facein_3node)
        # #################################################################################################
        left_cell_facein_duijiao = list(set(left_cell_facein_3node) - set(face_Node12))
        left_cell_facein_duijiao_x = Node_x[left_cell_facein_duijiao[0]-1]
        left_cell_facein_duijiao_y = Node_y[left_cell_facein_duijiao[0]-1]
        left_cell_duijiao_tans=Translation(left_cell_facein_duijiao_x,left_cell_facein_duijiao_y,train_point1_x,train_point1_y)
        left_cell_duijiao_Scaling = Scaling(train_point1_x,train_point1_y, train_point2_x,
                                            train_point2_y, left_cell_duijiao_tans[0], left_cell_duijiao_tans[1])  # 应该是0 0
        left_cell_duijiao_Rotation=Rotation(train_point2_Scaling[0],train_point2_Scaling[1],
                                            left_cell_duijiao_Scaling[0],left_cell_duijiao_Scaling[1])
        right_cell_facein=Right_Cell_Index[inputface-1]

        if right_cell_facein!=0:
            right_cell_facein_3node=[Cell_Node_Index[3*(right_cell_facein-1)],
                                     Cell_Node_Index[3*(right_cell_facein-1)+1],Cell_Node_Index[3*(right_cell_facein-1)+2]]
            right_cell_facein_duijiao = list(set(right_cell_facein_3node) - set(face_Node12))
            right_cell_facein_duijiao_x = Node_x[right_cell_facein_duijiao[0] - 1]
            right_cell_facein_duijiao_y = Node_y[right_cell_facein_duijiao[0] - 1]
        # 将以上两个对角点在以阵面为基准坐标中，找出在基准坐标上方的点，因为这样的才，，满足右手定则


            right_cell_duijiao_tans=Translation(right_cell_facein_duijiao_x,right_cell_facein_duijiao_y,train_point1_x,train_point1_y)
            right_cell_duijiao_Scaling = Scaling(train_point1_x,train_point1_y,
                                                 train_point2_x,train_point2_y, right_cell_duijiao_tans[0], right_cell_duijiao_tans[1])  # 应该是0 0
            right_cell_duijiao_Rotation=Rotation(train_point2_Scaling[0],
                                                 train_point2_Scaling[1],right_cell_duijiao_Scaling[0],right_cell_duijiao_Scaling[1])

        if left_cell_duijiao_Rotation[1]>0: # 将以上两个对角点在以阵面为基准坐标中，找出在基准坐标上方的点，因为这样的才，，满足右手定则
            new_point=left_cell_duijiao_Rotation
            new_point_old_zuobiao=left_cell_facein_duijiao[0]
        # if right_cell_duijiao_Rotation[1]>0:
        #     new_point=right_cell_duijiao_Rotation
        #     new_point_old_zuobiao = right_cell_facein_duijiao[0]
        #循环输入左边点的所有临点，然后依次输入右边点，注意所有边长小于阵面才可以取，然后合适的存入训练样本
        pp = Neighbourhood_Node(train_point1)  # 找出输入点相连的所有边
        Node_Face_Len = len(pp)
        temp12 = Node_Nearby_Node(Node_Face_Len, train_point1)  ##找出输入点相邻的所有点号，自己本身除外
        point1_nearby=temp12
        point1_nearby=list(set(point1_nearby)-set(face_Node12))

        temp13 = Node_zuobiao(len(point1_nearby))
        point1_nearby_zuobiao=temp13

        # 找所有的右边端点临点
        pp = Neighbourhood_Node(train_point2)  # 找出输入点相连的所有边
        Node_Face_Len = len(pp)
        temp12 = Node_Nearby_Node(Node_Face_Len, train_point2)  ##找出输入点相邻的所有点号，自己本身除外
        point2_nearby=temp12
        point2_nearby = list(set(point2_nearby) - set(face_Node12))
        temp13 = Node_zuobiao(len(point2_nearby))
        point2_nearby_zuobiao=temp13

        for i in range(len(point2_nearby)):
            node3_nearby.append([]) #第三个点的相邻点，因为第三个点不定，所以这里做了一个二维数值来存储第三个点以及第三个点的临点（也就是第四个点）
            node3=point2_nearby[i]
            pp=Neighbourhood_Node(node3)  # 找出输入点相连的所有边
            Node_Face_Len = len(pp)
            temp23 = Node_Nearby_Node(Node_Face_Len, node3)  ##找出输入点相邻的所有点号，自己本身除外
            point3_nearby = temp23
            point3_nearby = list(set(point3_nearby) - set(face_Node12))  #因为node2是node3的临点，但是在选择的时候不能选这点，所以要减去
            node3_nearby[i]=point3_nearby
            node3_nearby[i]=point3_nearby

        taindata=[]
        len1_2=round(len_calcu(train_point1_x,train_point1_y,train_point2_x,train_point2_y),4) #计算阵面的长度,取小数点后两位
     #从阵面的起点出发，搜索第一个点的所有相邻点
        for i in range(len(point1_nearby)):
            test1=point1_nearby[i]
            for j in range(len(point2_nearby)):
                test2=point2_nearby[j]
                tmep_len=len(node3_nearby[j])
                for k in range(len(node3_nearby[j])):
                    test3=node3_nearby[j][k]
                    test4=point2_nearby[j]
                    uuu=front_face2_in_boundary(train_point2, point2_nearby[j])
                    if front_face2_in_boundary(train_point2,point2_nearby[j])==1:
                        continue  #如果第二个阵面在边界上就不存取，因为没有办法提取标签
                    if point1_nearby[i]==9 and train_point2==11 and test4==10:
                        oo=1
                        node11=point1_nearby[i]
                        node12=train_point1
                        node21=train_point1
                        node22=train_point2

                        aotu_test=aotu_judge_func(node11, node12, node21, node22)

                    if point1_nearby[i] == node3_nearby[j][k]:
                        node11=point1_nearby[i]
                        node12=train_point1
                        node21=train_point1
                        node22=train_point2

                        aotu_test=aotu_judge_func(node11, node12, node21, node22)
                        if aotu_test<0:
                            continue #是出现了应该出现的四点组成四边形，而且还向外长的情况

                    if point1_nearby[i] != point2_nearby[j]:
                        taindata.append(point1_nearby[i]) #第一个引导点
                        taindata.append(train_point1)     #阵面左边点,第2个引导点
                        taindata.append(train_point2)     #阵面右边点,第3个引导点
                        taindata.append(point2_nearby[j]) #第4个引导点
                        taindata.append(node3_nearby[j][k])  # 第5个引导点
                        taindata.append(new_point_old_zuobiao)  #老坐标答案点
                        new_point_old_zuobiao2=duijiao_of_face(train_point2,point2_nearby[j])#
                        taindata.append(new_point_old_zuobiao2)  # 第二个阵面的答案点，需要后面去找，目前用999代替

                        test1=point1_nearby[i]
                        test2=train_point1
                        test3=train_point2
                        test4=point2_nearby[j]
                        test5=node3_nearby[j][k]
                        duijiao1=new_point_old_zuobiao
                        duijiao2=new_point_old_zuobiao2
                        test_all=[test1,test2,test3,test4,test5,duijiao1,duijiao2]
                        ijk=[i,j,k]
                        a=1
        number_zushu = int(len(taindata) / 7)  # 每7个数据是一组，包含2个阵面所含的3个点，两个引导点，2个答案
        ########################以下显示提取点数据####################################
        if display_flag == 1:

            # plt.figure(figsize=(11, 11), dpi=80)
            for i in range(number_zushu):  #
                plt.figure(figsize=(11, 11), dpi=80)

                for j in range(4): #每提取一次会出现三条边，一个答案点，通过此循环，每次生成一条边
                    temp_taindata=taindata[i*7+j]  # 提取1,2;2,3;3,4形成3条边
                    temp_taindata_next = taindata[i * 7 + j+1]
                    temp_taindata_x=[Node_x[temp_taindata-1],Node_x[temp_taindata_next-1]]
                    temp_taindata_y =[Node_y[temp_taindata - 1],Node_y[temp_taindata_next - 1]]
                    if j==2:
                        # 阵面2显示用的坐标
                        advance2_display_x = temp_taindata_x
                        advance2_display_y = temp_taindata_y
                        plt.plot(advance2_display_x, advance2_display_y, 'g-s', linewidth=5, color='red',
                                 markerfacecolor='red',
                                 marker='o')
                    else:
                        plt.plot(temp_taindata_x, temp_taindata_y, 'g-s', linewidth=5,color='blue', markerfacecolor='blue', marker='o')

                daan_x = Node_x[taindata[i * 7 + 5] - 1]  # 单独保存点的x数据，以便单独显示
                daan_y = Node_y[taindata[i * 7 + 5] - 1]  # 单独保存点的y数据，以便单独显示
                plt.scatter(daan_x, daan_y,s=80,c='r',marker="D") #打印答案点1
                daan_x = Node_x[taindata[i * 7 + 6] - 1]  # 单独保存点的x数据，以便单独显示
                daan_y = Node_y[taindata[i * 7 + 6] - 1]  # 单独保存点的y数据，以便单独显示
                plt.scatter(daan_x, daan_y,s=80,c='r',marker="D") #打印答案点1


                plt.plot(advance_display_x, advance_display_y, 'g-s', linewidth=5, color='red', markerfacecolor='red',
                         marker='o')
                left_cell_facein_3node_x=display_tri_feature1[0]
                left_cell_facein_3node_y=display_tri_feature1[1]
                left_cell_facein_3node_y2=display_tri_feature1[2]
                plt.fill_between(left_cell_facein_3node_x, left_cell_facein_3node_y, left_cell_facein_3node_y2,facecolor="lawngreen")

                #第二个三角形特征区域
                plt.plot(advance2_display_x, advance2_display_y, 'g-s', linewidth=5, color='red', markerfacecolor='red',
                         marker='o')
                node2=taindata[i*7+2] #从第0个点到第6个点共7个点，取第2个点，注意是从第0个开始技术，第二个也就是通常意义的第三个
                node3 = taindata[i * 7 + 3]
                node6 = taindata[i * 7 + 6]
                left_cell_facein_3node2=[node2,node3,node6]
                display_tri_feature2 = display_tri_feature(left_cell_facein_3node2)
                left_cell_facein_3node_x=display_tri_feature2[0]
                left_cell_facein_3node_y=display_tri_feature2[1]
                left_cell_facein_3node_y2=display_tri_feature2[2]

                plt.fill_between(left_cell_facein_3node_x, left_cell_facein_3node_y, left_cell_facein_3node_y2,facecolor="lawngreen")
                dispaly()
        #
        #     # plt.show()   #用于显示，为了加快仿真，采用上方的保存方式，免得出现后去关闭
        # plt.show()
        #以下代码是为了输出训练数据train_data_output：

        for i in range(number_zushu):  #某条边有多少组训练数据
            # # Node_Face.append([])
            # train_data_output.append([])
            new_node=[]
            test_data=[]
            for j in range(7):#取0到6共7个数
                # # def Translation(x, y, x0, y0):  # x,y表示要任何要平移的点，x0，y0表示将以何点为圆心作为参考点
                # def Scaling(x_start, y_start, x_end, y_end, x, y):  # 注意xy是经过平移的点
                # def Rotation(x_end, y_end, x_rot, y_rot):  # x_end,y_end为第一步平移后face另外一个点的值以便求与全局坐标系求夹角ceta，
                # # xrot yrot为任意需要旋转的坐标
                temp_taindata=taindata[i * 7 + j]
                new_node_trans=Translation(Node_x[temp_taindata-1], Node_y[temp_taindata-1],train_point1_x,train_point1_y)

                new_node_Scaling=Scaling(train_point1_x, train_point1_y, train_point2_x, train_point2_y,new_node_trans[0], new_node_trans[1])

                new_node_Rotation=Rotation(train_point2_Scaling[0], train_point2_Scaling[1],new_node_Scaling[0], new_node_Scaling[1])

                train_data_output.append(new_node_Rotation[0])
                train_data_output.append(new_node_Rotation[1])
                test_data.append(new_node_Rotation[0])
                test_data.append(new_node_Rotation[1])
            node0=taindata[i * 7 + 0];node1=taindata[i * 7 + 1];node2=taindata[i * 7 + 2];node3=taindata[i * 7 + 3];
            node4 = taindata[i * 7 + 4];node5 = taindata[i * 7 + 5];node6 = taindata[i * 7 + 6]
            p_test = [node0, node1, node2, node3, node4, node5, node6]
            if test_data[0]==0.61 and test_data[1]==-1.07 and test_data[6]==1.94 and test_data[7]==0.66 and test_data[8]==1.03 and test_data[9]==1.27:
                yyy=1
            if test_data[0]==0.32 and test_data[1]==0.88 and test_data[6]==1.68 and test_data[7]==0.51 and test_data[8]==2.39 and test_data[9]==1.11:
                yyy=1
            if test_data[0]==0.32 and test_data[1]==0.88 and test_data[6]==1.79 and test_data[7]==-0.53 and test_data[8]==1.68 and test_data[9]==0.51:
                yyy=1
            if test_data[0]==0.32 and test_data[1]==0.88 and test_data[6]==0.7 and test_data[7]==-0.73 and test_data[8]==1.41 and test_data[9]==-1.57:
                yyy=1
            if test_data[0]==-0.08 and test_data[1]==0.58 and test_data[6]==1.47 and test_data[7]==0.97 and test_data[8]==0.88 and test_data[9]==2.0:
                yyy=1




            print("p_test",p_test)
            if i==20:
                mo=1
            if (node0!=node5) and (node5!=node6) and (node6!=node4) and (node1!=node6) and (node5!=node3) and (node0!=node4):
                train_data_output.append(1)
                flag_1=flag_1+1
            elif (node0==node5) and (node5!=node6) and (node6!=node4) and (node1!=node6) and (node5!=node3) and (node0!=node4):
                train_data_output.append(2)
                flag_2 = flag_2 + 1
            elif (node0!=node5) and (node5!=node6) and (node6==node4) and (node1!=node6) and (node5!=node3) and (node0!=node4):
                train_data_output.append(3)
                flag_3 = flag_3 + 1
            elif (node0==node5) and (node5!=node6) and (node6==node4) and (node1!=node6) and (node5!=node3) and (node0!=node4):
                train_data_output.append(4)
                flag_4 = flag_4 + 1
            elif (node0!=node5) and (node5==node6) and (node6!=node4) and (node1!=node6) and (node5!=node3) and (node0!=node4):
                train_data_output.append(5)
                flag_5 = flag_5+ 1
            elif (node0==node5) and (node5==node6) and (node6!=node4) and (node1!=node6) and (node5!=node3) and (node0!=node4):
                train_data_output.append(6)
                flag_6 = flag_6 + 1
            elif (node0!=node5) and (node5==node6) and (node6==node4) and (node1!=node6) and (node5!=node3) and (node0!=node4):
                train_data_output.append(7)
                flag_7 = flag_7 + 1
            elif (node0==node5) and (node5==node6) and (node6==node4) and (node1!=node6) and (node5!=node3) and (node0==node4):
                train_data_output.append(8)
                flag_8 = flag_8 + 1
            elif (node0!=node5) and (node5!=node6) and (node6!=node4) and (node1==node6) and (node5==node3) and (node0!=node4):
                train_data_output.append(9)
                flag_9 = flag_9 + 1
            elif (node0!=node5) and (node5!=node6) and (node6!=node4) and (node1==node6) and (node5==node3) and (node0==node4):
                train_data_output.append(10)
                flag_10 = flag_10 + 1
            else:
                train_data_output.append(222)
                flag_error= flag_error + 1
                print ("error is ",i)
            print("medel",train_data_output[len(train_data_output)-1])
        number_data = 2 * 7 + 1  # 每一组多少个数据
        train_data_output_zushu=int(len(train_data_output)/number_data)  #每五个数据是一组，包含一个阵面所含的两个点，两个引导点，一个答案
# 以下代码for循环是显示归一化后的数据
        print("flag_1=",flag_1)
        print("flag2=", flag_2)
        print("flag3=", flag_3)
        print("flag4=", flag_4)
        print("flag5=", flag_5)
        print("flag6=", flag_6)
        print("flag7=", flag_7)
        print("flag8=", flag_8)
        print("flag9=", flag_9)
        print("flag10=", flag_10)
        print("flagerror=", flag_error)
        fig_flag2 = 0
        if display_flag==1:
            for i in range(train_data_output_zushu):  #
                if display_flag == 1:
                    plt.figure(figsize=(11, 11), dpi=80)
                for j in range(4): #每提取一次会出现三条边，一个答案点，通过此循环，每次生成一条边
                    temp_taindata_out_x=[]
                    temp_taindata_out_y=[]
                    temp_taindata_out_x.append(train_data_output[i * number_data + 2*j+0])  #每组有11个数，5个点的xy坐标加一个model
                    temp_taindata_out_x.append(train_data_output[i * number_data + 2 * j + 2])  #
                    temp_taindata_out_y.append(train_data_output[i * number_data + 2 * j + 1])  #
                    temp_taindata_out_y.append(train_data_output[i * number_data + 2 * j + 3])  #
                    plt.plot(temp_taindata_out_x, temp_taindata_out_y, 'g-s', linewidth=2,color='b', markerfacecolor='b', marker='o')
                daan_out_x = train_data_output[i * number_data + 2 * 5 + 0]  # 单独保存点的x数据，以便单独显示
                daan_out_y = train_data_output[i * number_data + 2 * 5 + 1]  # 单独保存点的y数据，以便单独显示
                plt.scatter(daan_out_x, daan_out_y,s=80,c='r',marker="D") #打印答案点

                daan_out_x = train_data_output[i * number_data + 2 * 6 + 0]  # 单独保存点的x数据，以便单独显示
                daan_out_y = train_data_output[i * number_data + 2 * 6 + 1]  # 单独保存点的y数据，以便单独显示
                plt.scatter(daan_out_x, daan_out_y,s=80,c='r',marker="D") #打印答案点
                # dispaly()

                plt.xlim((-3, 3))
                plt.xticks(np.linspace(-2, 3, 6, endpoint=True))
                plt.ylim((-3, 3))
                plt.yticks(np.linspace(-2, 3, 6, endpoint=True))

                fig_flag2 = fig_flag2 + 1
                if display_flag == 1:
                    plt.savefig(r"./train_fig/Chart_nomornize {}.png".format(fig_flag2),
                            dpi=80)  # 保存分辨率为80的图片

                    # plt.show()   #用于显示，为了加快仿真，采用上方的保存方式，免得出现后去关闭


    #     #以下代码是将归一化后的数据反归一化，以便验证归一化数据的正确性
    #     for i in range(train_data_output_zushu):
    #         plt.figure(figsize=(9, 9))
    #
    #         for j in range(0,3):#一共五个点需要转换,要显示一条边，需要有第一个点和下一个点的数据，以下是第一点数据
    #             old_x=[]
    #             old_y=[]
    #             old_node_rotation=Anti_Rotation(train_point2_Scaling[0], train_point2_Scaling[1],
    #                                             train_data_output[i * 11 + 2*j+0], train_data_output[i * 11 + 2*j+1])
    #             old_node_Scaling=Anti_Scaling(train_point1_x, train_point1_y, train_point2_x,
    #                                           train_point2_y,old_node_rotation[0], old_node_rotation[1])
    #             old_node_trans = Anti_Translation(old_node_Scaling[0], old_node_Scaling[1], train_point1_x,train_point1_y)
    #             old_x.append(old_node_trans[0])
    #             old_y.append(old_node_trans[1])
    #          #以下是第二点数据
    #             old_node_rotation2=Anti_Rotation(train_point2_Scaling[0], train_point2_Scaling[1],
    #                                             train_data_output[i * 11 + 2*j+2+0], train_data_output[i * 11 + 2*j+2+1])
    #             old_node_Scaling2=Anti_Scaling(train_point1_x, train_point1_y, train_point2_x,
    #                                           train_point2_y,old_node_rotation2[0], old_node_rotation2[1])
    #             old_node_trans2 = Anti_Translation(old_node_Scaling2[0], old_node_Scaling2[1], train_point1_x,train_point1_y)
    #
    #             old_x.append(old_node_trans2[0])
    #             old_y.append(old_node_trans2[1])
    #
    #             plt.plot(old_x, old_y, 'g-s', linewidth=2, color='b', markerfacecolor='b',marker='o')
    #
    #             # plt.xticks(fontsize=front_size)
    #             # plt.yticks(fontsize=front_size)
    #         old_daan_rotation=Anti_Rotation(train_point2_Scaling[0], train_point2_Scaling[1],
    #                                             train_data_output[i * 11 + 8], train_data_output[i * 11 + 9])
    #         old_daan_Scaling=Anti_Scaling(train_point1_x, train_point1_y, train_point2_x,
    #                                           train_point2_y,old_daan_rotation[0], old_daan_rotation[1])
    #         old_daan_trans=Anti_Translation(old_daan_Scaling[0], old_daan_Scaling[1], train_point1_x,train_point1_y)
    #         plt.scatter(old_daan_trans[0], old_daan_trans[1], s=80, c='r', marker="D")  # 打印答案点
    #
    #     plt.show()   #用于显示，为了加快仿真，采用上方的保存方式，免得出现后去关闭
    #
    # print "Node_x=", Node_x
    # print "Node_y=", Node_y
    # print "Face_Node_Number=",Face_Node_Number
    # print "Face_Node_Index=",Face_Node_Index
    # print "Left_Cell_Index=",Left_Cell_Index
    # print "Right_Cell_Index",Right_Cell_Index
    # x
    # print "Cell_Node_Number=",Cell_Node_Number
    # print "Cell_Node_Index=",Cell_Node_Index

    file = open('train_data_output4.txt', 'w')
    file.write(str(train_data_output));
    file.close()

    print ("GAME OVER！！！")















