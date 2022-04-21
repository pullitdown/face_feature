#coding=gbk
import cv2
from PySide2.QtWidgets import QApplication,QPushButton,QLabel,QMessageBox, QStackedWidget,QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QBrush,QPixmap,QPalette,QImage,QPixmap
from PySide2.QtCore  import QTimer
import PySide2
import time
import numpy as np
from PIL import Image
from numpy.__config__ import show
import dlib
from scipy import signal

from functools import partial

from pandas import cut
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from skimage import filters,io,color

##############################################################################################
#shetou
rect = [0, 0, 0, 0]  # 设定需要分割的图像范围
leftButtonDowm = False  # 鼠标左键按下
leftButtonUp = True  # 鼠标左键松开
def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    width_new = 1080
    height_new = 1080
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))

    return img_new

def cut_face(img):
    face_catch=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    face_pos=face_catch.detectMultiScale(img,1.3,5)
    if(len(face_pos)>1 or len(face_pos)==0):
        return img,0
    x,y,w,h=face_pos[0]
    return img[int(y+0.8*h):int(y+1.2*h),int(x+0.3*w):int(x+0.7*w)],1

def jun_hen(img):
    B,G,R = cv2.split(img) #获得8通道的图像
    EB=cv2.equalizeHist(B)
    EG=cv2.equalizeHist(G)
    ER=cv2.equalizeHist(R)
    equal=cv2.merge((EB,EG,ER))
    return equal




# 鼠标事件的回调函数
def on_mouse(event, x, y, flag, param):
    global rect
    global leftButtonDowm
    global leftButtonUp

    # 鼠标左键按下
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDowm = True
        leftButtonUp = False

    # 移动鼠标事件
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDowm and not leftButtonUp:
            rect[2] = x
            rect[3] = y

            # 鼠标左键松开
    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDowm and not leftButtonUp:
            x_min = min(rect[0], rect[2])
            y_min = min(rect[1], rect[3])

            x_max = max(rect[0], rect[2])
            y_max = max(rect[1], rect[3])

            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDowm = False
            leftButtonUp = True


#需要选中的
def cut_she(img):
    global rect
    global leftButtonDowm
    global leftButtonUp
    mask = np.zeros(img.shape[:2], np.uint8)
    SIZE = (1, 65)
    # print(img.shape[0])
    # print(img.shape[1])
    # 变换图像通道bgr->rgb
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #分割
    bgdModle = np.zeros(SIZE, np.float64)
    fgdModle = np.zeros(SIZE, np.float64)

    cv2.namedWindow('img')  # 指定窗口名来创建窗口
    cv2.setMouseCallback('img', on_mouse)  # 设置鼠标事件回调函数 来获取鼠标输入
    cv2.imshow('img', img)  # 显示图片
    while cv2.waitKey(2) == -1:
    # 左键按下，画矩阵
        if leftButtonDowm and not leftButtonUp:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow('img', img_copy)

        # 左键松开，矩形画好
        elif not leftButtonDowm and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
            rect[2] = rect[2] - rect[0]
            rect[3] = rect[3] - rect[1]
            rect_copy = tuple(rect.copy())
            #rect = [0, 0, 0, 0]
            # 物体分割
            cv2.grabCut(img, mask, rect_copy, bgdModle, fgdModle, 8, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img_show = img * mask2[:, :, np.newaxis]
            # # 显示图片分割后结果--显示原图
            # cv2.imshow('grabcut', img_show)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #print(rect_copy)
            img_show = img_show[rect_copy[1]:rect_copy[1]+rect_copy[3],rect_copy[0]:rect_copy[0]+rect_copy[2]]
            #x, y, w, h = rect_copy
            #img_show = img_show[y:y+h,x:x+w]

            return img_show


# #原来的
# def cut_she(img):
#     mask = np.zeros(img.shape[:2], np.uint8)
#     SIZE = (1, 65)
#     # print(img.shape[0])
#     # print(img.shape[1])
#     # 变换图像通道bgr->rgb
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#     #分割
#     bgdModle = np.zeros(SIZE, np.float64)
#     fgdModle = np.zeros(SIZE, np.float64)
#     rect = (int((img.shape[0])/10), int((img.shape[1])/7), int((img.shape[0])*3/5), int((img.shape[1])*2/3))
#     for i in range(3):
#         cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 11, cv2.GC_INIT_WITH_RECT)
#         mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#         img *= mask2[:, :, np.newaxis]
#         cv2.grabCut(img, mask2, rect, bgdModle, fgdModle, 1)
#         img *= mask2[:, :, np.newaxis]
#     return img

def kmeans(img):
    img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 聚类,这里k=3;有黑色的部分未处理提取
    compactness, labels, centers = cv2.kmeans(img_flat, 4, None, criteria, 10, flags)

    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))

    return centers,img_output

def find(centers):
    centers_R = np.empty(shape = 4)
    ind = 0
    for i in centers:
        centers_R[ind] = i[2]
        ind = ind+1

    shezhi_index = np.argsort(centers_R)[-2]
    shetai_index = np.argsort(centers_R)[-1]
    return shezhi_index,shetai_index

def tai_fea(img_output,img,shetai_index):
    mask4 = np.where(img_output==shetai_index,255,0)
    mask4 = np.uint8(mask4)
    img_shetai = cv2.bitwise_and(img, img, mask=mask4)
    #转Lab和HSL颜色空间,区分舌苔
    img_lab = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2LAB)
    img_hsl = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2HLS)
    cv2.imshow('shetai',img_shetai)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # plt.subplot(121), plt.imshow(img), plt.title('input')
    # plt.subplot(122), plt.imshow(img_shetai), plt.title('fin')
    # plt.show()
    #舌苔亮度值
    shetai_H,shetai_L1,shetai_S = cv2.split(img_hsl)
    exist = (shetai_L1 != 0)
    shetai_L_meanvalue = shetai_L1.sum()/exist.sum()

    #舌苔a,b值
    shetai_L2,shetai_A,shetai_B = cv2.split(img_lab)
    exist = (shetai_A != 0)
    shetai_A_meanvalue = shetai_A.sum()/exist.sum()


    exist = (shetai_B != 0)
    shetai_B_meanvalue = shetai_B.sum()/exist.sum()
    return shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue

def zhi_fea(img_output,img,shezhi_index):
    mask5 = np.where(img_output==shezhi_index,255,0)
    mask5 = np.uint8(mask5)
    img_shezhi = cv2.bitwise_and(img, img, mask=mask5)
    cv2.imshow('shezhi',img_shezhi)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # plt.subplot(121), plt.imshow(img), plt.title('input')
    # plt.subplot(122), plt.imshow(img_shezhi), plt.title('fin')
    # plt.show()
    #舌质分析
    shezhi_R, shezhi_G, shezhi_B= cv2.split(img_shezhi)
    exist = (shezhi_R != 0)
    shezhi_R_meanvalue = shezhi_R.sum()/exist.sum()
    return shezhi_R_meanvalue

#原图

def analyze():
    return 0

##############################################################################################
def showimg(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def showbignum(name,img):
    print(name,"\n:\n",img)
    cv2.normalize(img,img,0,255,norm_type=cv2.NORM_MINMAX)
    img = cv2.convertScaleAbs(img)
    print(name,"\n:\n",img)

def kernelbysize(size):
    kernel_size=size
    kernel_size_half=int(kernel_size/2)
    #腐蚀和膨胀,首先获得腐蚀和膨胀操作的kernel
    kernel=np.zeros((kernel_size,kernel_size),np.uint8)
    for lenght in range(kernel_size_half):
        for x in range(kernel_size):
            for y in range(kernel_size):
                if (x-kernel_size_half)*(x-kernel_size_half)+(y-kernel_size_half)*(y-kernel_size_half)<=lenght*lenght:
                    kernel[x,y]+=int(255/kernel_size)+2

class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        self.capIsOpen=0
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        # self.ui = QUiLoader().load('./qt/ui/untitled.ui')

        # self.ui.button.clicked.connect(self.handleCalc)S

        self.mainPage=QUiLoader().load('./fea_all/ui/main.ui')#load主页面
        # palette = QPalette()
        # # showimg('kkl',cv2.imread(".\qt\img\pizhi.jpg"))
        # k=QPixmap().load(".\qt\img\pizhi.jpg")
        # palette.setBrush(self.mainPage.backgroundRole(), QBrush(k)) #背景图片
        # # palette.setBrush(QPalette.Background, QBrush(icon))
        # self.mainPage.setPalette(palette)
        self.display_video_stream(cv2.imread(".\\fea_all\img\logo.jpg"),self.mainPage.logo)
        self.mainPage.setWindowTitle("中医养生建议系统demo-1.0")
        self.hasimg=0






        self.ui = QUiLoader().load('./fea_all/ui/imgpage.ui')
        self.ui_2=QUiLoader().load('./fea_all/ui/quespage.ui')
        self.ui_3=QUiLoader().load('./fea_all/ui/goal.ui')
        self.face_catch=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

        self.eye_catch=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
        #self.ui = QUiLoader().load('./qt/ui/untitled.ui')

        self.ui.toolButton.clicked.connect(self.start)
        self.ui.toolButton_6.clicked.connect(self.xuanze)
        self.ui.toolButton_4.clicked.connect(self.pulse_feature)
        self.ui.toolButton_2.clicked.connect(partial(self.faceFeature,0))
        self.ui.toolButton_5.clicked.connect(self.nextPage)
        self.ui_2.toolButton_5.clicked.connect(self.question)
        self.ui_2.toolButton_6.clicked.connect(self.restart)
        self.ui.toolButton_3.clicked.connect(self.shetou_fea)
        self.timer_Active = 0
        self.timer = QTimer()
        self.timer.start()            # 实时刷新，不然视频不动态
        self.timer.setInterval(100)   # 设置刷新时间
        self.timer_4 = QTimer()
         # 实时刷新，不然视频不动态
        self.bpm_list = []
        self.timer_4.setInterval(100)  # 设置刷新时间
        self.timer_4.start()
        self.questions=[["1、近期是否有特别怕冷或者怕热的情况",("A.是","B.否")],["2、有没有出现手心、脚心、胸中发热的情况",("A.有","B.无明显症状")],
       ["3、近期的出汗状况？",("A.无明显症状","B.睡觉时易出汗","C.日常生活汗多","D.有出冷汗的现象")],
       ["4、近期是否有经常出现头晕或者头痛的症状",("A.是","B.否")],["5、近期是否有身体乏力，精神不振的感觉？",("A.是","B.否")],
       ["6、近期是否有出现腰酸背痛，精力不足的症状",("A.是","B.否")],
       ["7、近期大便的症状",("A.正常量且正常态(成型)","B.量少正常态（成型）","C.正常量，但大便粘滞","D.正常量，但大便稀","E.量多,腹泻")],
       ["8、近期胃口",("A.正常量","B.吃得少，无胃口","C.厌食，胃胀，泛酸","D.不想吃油腻的东西，容易对其产生呕吐感","容易饿，且多饮多尿")],
       ["9、近期的小便的症状",("A.正常量，且正常态","B.小便短黄","C.小便清长","D.小便有异味")],
       ["10、有无出现胸闷，心悸，胁胀和腹胀的症状",("A.有","B.无")],
       ["11、有无出现耳鸣和听力下降的问题",("A.有","B.无")],
       ["12、有无经常出现口干或者口渴的症状",("A.有","B.无")],
       ["13、平时多喝冷饮，热饮还是常温？",("A.冷饮","B.热饮","C.常温")],
       ["14、有无常病史",("A.糖尿病","B.脂肪肝","C.高血压","D.高血脂","E.高尿酸","F.低血压")]]
        self.nowQuestionIndex=0
        self.questionsLen=len(self.questions)
        self.ui_2.textBrowser.setText("点击开始答题,本文本框会出现问题,根据实际情况在以下选择框内选择一个答案")
        self.answerTable=[]#记录问卷选择的答案1
        self.isAnswer=1
        self.singleAnswer=[]#记录单个问题的回答,可以多选
        self.shetou_fea_data=[]
        self.display_video_stream(cv2.imread(".\\fea_all\img\capBackground.jpg"),self.ui.label)

        self.stack=QStackedWidget(self.mainPage)
        self.stack.addWidget(self.ui)
        self.stack.addWidget(self.ui_2)
        self.stack.addWidget(self.ui_3)
        self.mainPage.playout.addWidget(self.stack)

        self.mainPage.listWidget.insertItem(0,'图像提取部分')
        self.mainPage.listWidget.insertItem(1,'近况问答部分')
        self.mainPage.listWidget.insertItem(2,'结果展示')
        self.mainPage.listWidget.currentRowChanged.connect(self.showPage)


    def shetou_fea(self):
        #取得脸部下半
        img=self.img
        # img,flag = cut_face(img)
        # if flag == 0:
        #     QMessageBox.information(self.ui,"提示","目前背景环境不佳,或有多个人脸在检测区域内,请重试")
        #     return
        #处理大小
        img = img_resize(img)
        #均衡化
        equal = jun_hen(img)
        #采用中值滤波
        median = cv2.medianBlur(equal,5)
        #切割舌体
        img = cut_she(median)


        self.tougueimg=img
        showimg("all",self.tougueimg)
######################################################


        #Kmeans
        centers,img_output = kmeans(img)
        print(centers)
        #showimg("P2",img_output)
        #找点位
        shezhi_index,shetai_index = find(centers)
        #舌苔分析
        shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue = tai_fea(img_output,img,shetai_index)
        #舌质分析
        shezhi_R_meanvalue = zhi_fea(img_output,img,shezhi_index)
        self.shetou_fea_data.extend([shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue,shezhi_R_meanvalue])
        self.ui.textEdit.setText(str(shetai_L_meanvalue))

    def xuanze(self):#输入模型权重
        (fileName1, filetype)= QFileDialog.getOpenFileName()
        self.img = cv2.imread(fileName1)
        self.hasimg=1
        print(fileName1)
        self.display_video_stream(self.img,self.ui.label)


    def nextPage(self):
        self.stack.setCurrentIndex(1)
        #######################################################################
        self.evaluate()
        #########################################################################

    def showPage(self,i):
        self.stack.setCurrentIndex(i)




    def display_video_stream(self,frame,label):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)

        label.setPixmap(QPixmap.fromImage(image).scaled(label.width(), label.height()))

    def restart(self,event):
        self.nowQuestionIndex=0
        self.isAnswer=1
        self.answerTable=[]
        self.singleAnswer=[]
        self.ui_2.toolButton_5.setText("开始答题")
        while self.ui_2.select_table.count():
            widget=self.ui_2.select_table.takeAt(0).widget()
            widget.deleteLater()
        self.ui_2.textBrowser.setText("点击开始答题,本文本框会出现问题,根据实际情况在以下选择框内选择一个答案")

    def evaluate(self):
        self.sum_vector=np.zeros((9,))
        if self.face_vector[2]-self.face_vector[0]>6:
            self.sum_vector[0]+=1
            if self.face_vector.sum()>40:
                self.sum_vector[7]+=1
        if self.face_vector.sum()<20:
            self.sum_vector[7]+=1


        self.tongue_vector=[]

        self.status=["阴虚","阳虚","气虚","平和质","气郁","湿热","痰湿","血瘀","平和质"]

        self.answer_vector=[
            [[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]],
            [[1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]],
            [[0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0]],
            [[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]],
            [[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]],
            [[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]],
            [[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1]],
            [[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1]],
            [[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,1]],
            [[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1]],
            [[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1]],
            [[0,0,0,0,0,0,0,0,1],[1,0,0,0,0,1,0,0,0]],
            [[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1]],
            [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
        ]

        for idx,i in enumerate(self.answerTable):
            for j in i:
                self.sum_vector=self.sum_vector+np.array(self.answer_vector[idx][j])
        print(self.sum_vector)

        sta=np.argsort(self.sum_vector[:-1])[-2:]
        if self.sum_vector[-1]>8:
            sta[0]=8
        self.bodyinf=[]
        for i in sta:
            self.bodyinf.append(self.status[i])





    def question(self,event):
        if self.isAnswer==0:
            QMessageBox.information(self.ui_2,"提示","请点击选项再回答下一题")
            return
        self.isAnswer=0#设置为未回答
        if not self.singleAnswer ==[]:
            self.answerTable.append(self.singleAnswer)
        self.singleAnswer=[]
        i=self.nowQuestionIndex
        self.nowQuestionIndex+=1
        if i==0:
            self.ui_2.toolButton_5.setText("下一题")
        if i==self.questionsLen-1:
            self.ui_2.toolButton_5.setText("结束答题")
        if i>self.questionsLen:
            QMessageBox.information(self.ui_2,"提示","已经结束答题,如果想重试答题,请按重新作答键")
            return
        if i==self.questionsLen:
            self.stack.setCurrentIndex(2)


            self.evaluate()
            self.ui_3.adviseinf.setText(str(self.answerTable))
            self.ui_3.bodyinf.setText(str(self.bodyinf))
            self.display_video_stream(self.faceimg,self.ui_3.faceimg)
            self.display_video_stream(self.tougueimg,self.ui_3.tougueimg)
            self.display_video_stream(self.rangeimg,self.ui_3.rangeimg)
            #QMessageBox.information(self.ui_2,"检测结果",str(self.answerTable)+"您非常健康!")
            return

        self.ui_2.textBrowser.setText(self.questions[i][0])
        lenOfAnswer=len(self.questions[i][1])
        while self.ui_2.select_table.count():
            widget=self.ui_2.select_table.takeAt(0).widget()

            widget.deleteLater()
        for row,j in enumerate(range(0,lenOfAnswer,3)):
            for col in range(min(3,lenOfAnswer-j)):
                button = QPushButton(self.questions[i][1][j+col], self.ui_2)
                button.clicked.connect(partial(self.answerTable_append,button,j+col))
                self.ui_2.select_table.addWidget(button,row,col)




    def answerTable_append(self,widget ,num):
        self.isAnswer=1
        widget.setStyleSheet("QPushButton{color:black;background-color:rgb(51,123,4)}")
        self.singleAnswer.append(num)


    def start(self,event):
        if self.capIsOpen==0:
            self.capIsOpen=1
            self.ui.toolButton.setText("视频关闭")
            self.cap = cv2.VideoCapture(0)
            self.timer.timeout.connect(self.capPicture)
        else:
            self.ui.toolButton.setText("视频开启")
            self.hasimg=0
            self.display_video_stream(cv2.imread(".\\fea_all\img\capBackground.jpg"),self.ui.label)
            self.capIsOpen=0
            self.cap.release()
            self.ui.label.setText(" ")



    def faceFeature(self,test=1):#脸色提取
        all_color_df=np.zeros((3))
        df_num=0

        for ll in range(10):
            if(self.cap.isOpened() or self.hasimg):
                start,second,three=35,6,3#设定参数,分别是腐蚀/膨胀操作的kernel大小,腐蚀次数,膨胀次数
                img = self.img #得到当前的照片
                face_pos=self.face_catch.detectMultiScale(img,1.3,5)
                if(len(face_pos)>1 or len(face_pos)==0):
                  #  QMessageBox.information(self.ui,"提示","目前背景环境不佳,或有多个人脸在检测区域内,请重试")
                    continue
                x,y,w,h=face_pos[0]
                img=img[y:y+h,x:x+w]#人脸区域的图片
                Y,CR,CB=cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb))
                cv2.normalize(CR,CR,start,255,norm_type=cv2.NORM_MINMAX)
                cv2.normalize(CB,CB,start,255,norm_type=cv2.NORM_MINMAX)
                CR_array,CB_array=np.array(CR,dtype=np.float32),np.array(CB,dtype=np.float32)
                CR2=CR_array*CR_array
                CRB=CR_array/CB_array
                cv2.normalize(CR2,CR2,start,255,norm_type=cv2.NORM_MINMAX)
                cv2.normalize(CRB,CRB,start,255,norm_type=cv2.NORM_MINMAX)
                CR2,CRB=CR2.astype(np.float32),CRB.astype(np.float32)
                n=0.95*sum(CR2)/sum(CRB)
                t=(CR2-n*CRB)
                p1=CR2*t*t

                cv2.normalize(p1,p1,start,255,norm_type=cv2.NORM_MINMAX)

                p1=p1.astype(np.uint8)
                print(test)
                if test == 1:

                    showimg("CR", CR)
                    showimg("CB", CB)
                    showimg("CR_arr",CR_array)
                    showimg("CB_arr",CB_array)
                    showimg("CR2",CR2)
                    showimg("CRB",CRB)
                    showimg("P1",p1)

                kernel=kernelbysize(int(np.floor(img.shape[0]/5)))
                p1=cv2.erode(p1,kernel,iterations=second)#腐蚀
                #showimg("P1",p1)#35 6
                p1=cv2.dilate(p1,kernel,iterations=three)#膨胀
                #showimg("P1",p1)#35 6
                cv2.normalize(p1,p1,0,255,norm_type=cv2.NORM_MINMAX)

                p13=p1
                p13=255-p13

                con,hie=cv2.findContours(p13,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                p3_copy=p13.copy()

                if len(con)==1:
                   # QMessageBox.information(self.ui,"提示","目前背景环境不佳,或有多个人脸在检测区域内,请重试")
                    continue
                xm,ym,wm,hm= cv2.boundingRect(con[1])

                # mouthimg=img.copy()[xm:xm+wm,ym:ym+hm]
                # cha=cv2.cvtColor(mouthimg,cv2.COLOR_BGR2GRAY)
                # mask =cv2.threshold(cha,120,255,cv2.THRESH_TOZERO)
                # mask=cv2.cvtColor(mouthimg,cv2.COLOR_GRAY2BGR)
                # showimg("dd",mask)
                # self.mouthstatus={}
                # for i,col in enumerate(['b','g','r']):
                #     hist_mask0=cv2.calcHist([mouthimg],[i],mask,[25],[0,256])
                #     self.mouthstatus[i]+=np.argmax(hist_mask0)



                eye_pos=self.eye_catch.detectMultiScale(img)
                if len(eye_pos)!=2:
                    continue
                x0,y0,w0,h0=eye_pos[0]
                x1,y1,w1,h1=eye_pos[1]
                fin=img.copy()
                rect=cv2.rectangle(fin,(xm,ym),(xm+wm,ym+hm),(0,255,0),3)
                rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
                rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
                if test==1:
                    showimg("P1",p1)#35 6 3
                    showimg("p13",p13)
                    showimg('rect', rect)
                k=1
                if x0<xm:
                    k=-1
                first_face=(int(x0+k*0.3*w0),y0+h0,w0,int(ym-img.shape[1]/15-y0-h0))
                k=1
                if x1<xm:
                    k=-1
                second_face=(int(x1+k*0.3*w1),y1+h1,w1,int(ym-img.shape[1]/15-y1-h1))
                x0,y0,w0,h0=first_face
                x1,y1,w1,h1=second_face
                rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
                rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
                if test==1:
                    showimg('rect', rect)
                mask0=np.zeros(img.shape[:2],np.uint8)

                mask0[y0:y0+h0,x0:x0+w0]=255
                mask1=np.zeros(img.shape[:2],np.uint8)
                mask1[y1:y1+h1,x1:x1+w1]=255
                fin=img.copy()
                fin=cv2.cvtColor(fin,cv2.COLOR_BGR2GRAY)
                ret2,fin = cv2.threshold(fin,50,255,cv2.THRESH_BINARY)
                #fin =cv2.adaptiveThreshold(fin,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                #cv2.THRESH_BINARY,11,2)
                mask0=cv2.bitwise_and(mask0,fin,mask0)
                mask1=cv2.bitwise_and(mask1,fin,mask1)

                showmask=np.zeros(img.shape[:2],np.uint8)
                showmask=cv2.bitwise_or(mask1,mask0,showmask)
                self.mm=showmask
                # ret2,mask1 = cv2.threshold(mask1,200,255,cv2.THRESH_BINARY)
                # ret2,mask0 = cv2.threshold(mask0,200,255,cv2.THRESH_BINARY)
                # showimg('sb', mask0)
                # showimg('sb', mask1)
                rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
                rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)


                if test==1:
                    showimg('face_rect', rect)
                color_df=np.zeros((3))
                light=0
                for i,col in enumerate(['b','g','r']):
                    hist_mask0=cv2.calcHist([img],[i],mask0,[25],[0,256])
                    color_df[i]+=np.argmax(hist_mask0)
                    hist_mask1=cv2.calcHist([img],[i],mask1,[25],[0,256])
                    color_df[i]+=np.argmax(hist_mask1)
                color_df=color_df/2
                if  color_df[0]>0 and color_df[1]>0 and color_df[2]>0:
                    all_color_df+=color_df

                    rect=cv2.rectangle(img,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
                    rect=cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
                    self.faceimg=rect
                    self.showmask=self.mm
                    df_num+=1
            if df_num==0:
                df_num+=1
            self.face_vector=all_color_df/df_num#2
            self.ui.textEdit.setText(str([(k,i) for k,i in zip(['蓝','绿','红'],self.face_vector)]))


### 脉搏提取部分开始 ###

    # 获取前额的区域
    def get_forehead_roi(self, face_points):
        # 将这些点存储在Numpy数组中，可以很容易地通过切片得到x和y的最小值和最大值
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)
        # 前额两眉间区域
        min_x = int(points[21, 0])
        min_y = int(min(points[21, 1], points[22, 1]))
        max_x = int(points[22, 0])
        max_y = int(max(points[21, 1], points[22, 1]))
        left = min_x
        right = max_x
        top = min_y - (max_x - min_x)
        bottom = max_y * 0.98
        return int(left), int(right), int(top), int(bottom)

    # 获取鼻子的区域
    def get_nose_roi(self, face_points):
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)
        # 鼻子和脸颊
        min_x = int(points[36, 0])
        min_y = int(points[28, 1])
        max_x = int(points[45, 0])
        max_y = int(points[33, 1])
        left = min_x
        right = max_x
        top = min_y + (min_y * 0.02)
        bottom = max_y + (max_y * 0.02)
        return int(left), int(right), int(top), int(bottom)

    # 对两个像素数组的绿色值取平均值
    def get_avg(self, roi1, roi2):
        roi1_green = roi1[:, :, 1]
        roi2_green = roi2[:, :, 1]
        avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
        return avg

    # 获取感兴趣区域的绿色像素平均值，并在周围画一个绿色的矩形
    def get_roi_avg(self, frame, face_points):  # 照片，脸部关键点
        # 得到感兴趣的区域
        fh_left, fh_right, fh_top, fh_bottom = self.get_forehead_roi(face_points)
        nose_left, nose_right, nose_top, nose_bottom = self.get_nose_roi(face_points)
        # 将感兴趣的区域(ROI)平均分割
        fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
        nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
        return self.get_avg(fh_roi, nose_roi)

    # 滑动窗口贬低
    def sliding_window_demean(self, signal_values, num_windows):
        window_size = int(round(len(signal_values) / num_windows))
        demeaned = np.zeros(signal_values.shape)
        for i in range(0, len(signal_values), window_size):
            if i + window_size > len(signal_values):
                window_size = len(signal_values) - i
            curr_slice = signal_values[i: i + window_size]
            demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
        return demeaned

    # 创建指定的Butterworth过滤器并应用
    def butterworth_filter(self, data, low, high, sample_rate, order):
        nyquist_rate = sample_rate * 0.5
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)

    # 过滤信号数据
    def filter_signal_data(self, fps):  # 平均绿色像素数组，帧数
        # 确保数组没有无限或NaN值
        values = np.array(self.roi_avg_values)
        np.nan_to_num(values, copy=False)
        # 通过消除趋势和贬低来使信号平滑
        detrended = signal.detrend(values, type='linear')  # 从信号中删除线性趋势
        demeaned = self.sliding_window_demean(detrended, 15)  # 滑动窗口贬低
        # 用butterworth带通滤波器对信号进行滤波
        filtered = self.butterworth_filter(demeaned, self.MIN_HZ, self.MAX_HZ, fps, 5)
        return filtered


    # 计算每分钟的脉搏(BPM)
    def compute_bpm(self, filtered_values, fps, buffer_size):  # 滤波后数据，帧数，缓冲区大小，上次bpm
        # 快速傅里叶变换计算FFT
        fft = np.abs(np.fft.rfft(filtered_values))
        # 生成与FFT值相对应的频率列表
        fft_freq = fps / buffer_size * np.arange(buffer_size / 2 + 1)
        # 滤除FFT中不在[MIN_HZ, MAX_HZ]范围内的全部峰值，因为无BPM值可对应它们
        while True:
            max_idx = fft.argmax()
            bps = fft_freq[max_idx]
            if bps < self.MIN_HZ or bps > self.MAX_HZ:
                fft[max_idx] = 0
            else:
                bpm = bps * 60.0
                break
        # 在样本之间，心率变化不可能超过10%，所以使用加权平均来使BPM与最后一个BPM平滑
        if self.last_bpm > 0:
            bpm = (self.last_bpm * 0.9) + (bpm * 0.1)
        return bpm

    # 脉搏提取器
    def pulse_observer(self):
        frame = self.img
        # 如果无法从网络摄像头读取 ret_val == False
        if not self.ret_val:
            print("ERROR:  Unable to read from webcam.  Was the webcam disconnected?  Exiting.")
            return
        # 在画框之前先把画框复制一份，我们将在GUI中显示副本，原始帧将用于计算心率
        view = np.array(frame)
        # 心率图占窗口宽度的75%，BPM获得25%
        if self.graph_width == 0:
            self.graph_width = int(view.shape[1] * 0.75)
        if self.bpm_display_width == 0:
            self.bpm_display_width = view.shape[1] - self.graph_width
        # 使用dlib检测人脸
        faces = self.detector(frame, 0)
        if len(faces) == 1:
            face_points = self.predictor(frame, faces[0])
            roi_avg = self.get_roi_avg(frame, face_points)
            self.roi_avg_values.append(roi_avg)
            self.times.append(time.time())
            # Buffer已经满了，从顶部弹出值来删除它
            if len(self.times) > self.BUFFER_MAX_SIZE:
                self.roi_avg_values.pop(0)
                self.times.pop(0)
            curr_buffer_size = len(self.times)
            # 在有最小帧数之前，不要计算脉搏
            if curr_buffer_size > self.MIN_FRAMES:
                # 计算相关的次数
                time_elapsed = self.times[-1] - self.times[0]
                fps = curr_buffer_size / time_elapsed  # 帧数
                # 清理信号数据
                filtered = self.filter_signal_data(fps)
                self.graph_values.append(filtered[-1])
                if len(self.graph_values) > self.MAX_VALUES_TO_GRAPH:
                    self.graph_values.pop(0)
                # 计算并显示BPM
                bpm = self.compute_bpm(filtered, fps, curr_buffer_size)
                self.bpm_list.append(bpm)
                if len(self.bpm_list) > self.BPM_MAX_SIZE:
                    self.bpm_list.pop(0)
                self.last_bpm = bpm
                self.ui.textEdit.setText(str(bpm))
            else:
                # 如果没有足够的数据来计算脉搏，则显示一个带有加载文本和BPM占位符的空图
                pct = int(round(float(curr_buffer_size) / self.MIN_FRAMES * 100.0))
                self.ui.textEdit.setText('Computing pulse: ' + str(pct) + '%')

        else:
            # 没有检测到脸，所以必须清除值和时间列表，否则，当再次检测到人脸时，时间就会出现空白。
            del self.roi_avg_values[:]
            del self.times[:]

    def pulse_feature(self, event):
        self.BUFFER_MAX_SIZE = 500  # 存储的近期ROI平均值的数量
        self.MAX_VALUES_TO_GRAPH = 50  # 在脉冲图中显示最近的ROI平均值
        self.BPM_MAX_SIZE = 10
        self.MIN_HZ = 0.83  # 50 BPM - 最小允许心率
        self.MAX_HZ = 3.33  # 200 BPM - 最大允许心率
        self.MIN_FRAMES = 100  # 在计算心率之前所需的最小帧数
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./fea_all/data/shape_predictor_68_face_landmarks.dat')
        self.roi_avg_values = []
        self.graph_values = []
        self.times = []

        self.last_bpm = 0
        self.graph_height = 200
        self.graph_width = 0
        self.bpm_display_width = 0


        if self.capIsOpen==1 :
            if self.timer_Active==0:
                self.bpm_list = []
                self.ui.toolButton_4.setText("关闭心率采集")
                self.timer_Active = 1
                self.timer_4.timeout.connect(self.pulse_observer)

            else:
                self.timer_4.timeout.disconnect(self.pulse_observer)
                self.ui.toolButton_4.setText("心率采集")
                self.timer_Active = 0
        else:
            QMessageBox.information(self.ui,"提示","摄像头未开启,请重试")
            return





### 脉搏提取部分结束 ###


### 获取实时摄像头照片以及展示 ###
    def capPicture(self):
        if self.capIsOpen==1:#如果是开启视频

            # get a frame
            ret, img = self.cap.read()#读取摄像头
            self.ret_val = ret
            self.img = img
            self.last=self.img#上一次的图片
            height, width, bytesPerComponent = img.shape
            bytesPerLine = bytesPerComponent * width
            # 变换彩色空间顺序
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 转为QImage对象

            self.image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(self.image).scaled(self.ui.label.width(), self.ui.label.height()))








app = QApplication([])
stats = Stats()
stats.mainPage.show()
app.exec_()