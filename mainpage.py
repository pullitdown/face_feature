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
rect = [0, 0, 0, 0]  # �趨��Ҫ�ָ��ͼ��Χ
leftButtonDowm = False  # ����������
leftButtonUp = True  # �������ɿ�
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
    B,G,R = cv2.split(img) #���8ͨ����ͼ��
    EB=cv2.equalizeHist(B)
    EG=cv2.equalizeHist(G)
    ER=cv2.equalizeHist(R)
    equal=cv2.merge((EB,EG,ER))
    return equal




# ����¼��Ļص�����
def on_mouse(event, x, y, flag, param):
    global rect
    global leftButtonDowm
    global leftButtonUp

    # ����������
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDowm = True
        leftButtonUp = False

    # �ƶ�����¼�
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDowm and not leftButtonUp:
            rect[2] = x
            rect[3] = y

            # �������ɿ�
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


#��Ҫѡ�е�
def cut_she(img):
    global rect
    global leftButtonDowm
    global leftButtonUp
    mask = np.zeros(img.shape[:2], np.uint8)
    SIZE = (1, 65)
    # print(img.shape[0])
    # print(img.shape[1])
    # �任ͼ��ͨ��bgr->rgb
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #�ָ�
    bgdModle = np.zeros(SIZE, np.float64)
    fgdModle = np.zeros(SIZE, np.float64)

    cv2.namedWindow('img')  # ָ������������������
    cv2.setMouseCallback('img', on_mouse)  # ��������¼��ص����� ����ȡ�������
    cv2.imshow('img', img)  # ��ʾͼƬ
    while cv2.waitKey(2) == -1:
    # ������£�������
        if leftButtonDowm and not leftButtonUp:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow('img', img_copy)

        # ����ɿ������λ���
        elif not leftButtonDowm and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
            rect[2] = rect[2] - rect[0]
            rect[3] = rect[3] - rect[1]
            rect_copy = tuple(rect.copy())
            #rect = [0, 0, 0, 0]
            # ����ָ�
            cv2.grabCut(img, mask, rect_copy, bgdModle, fgdModle, 8, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img_show = img * mask2[:, :, np.newaxis]
            # # ��ʾͼƬ�ָ����--��ʾԭͼ
            # cv2.imshow('grabcut', img_show)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #print(rect_copy)
            img_show = img_show[rect_copy[1]:rect_copy[1]+rect_copy[3],rect_copy[0]:rect_copy[0]+rect_copy[2]]
            #x, y, w, h = rect_copy
            #img_show = img_show[y:y+h,x:x+w]

            return img_show


# #ԭ����
# def cut_she(img):
#     mask = np.zeros(img.shape[:2], np.uint8)
#     SIZE = (1, 65)
#     # print(img.shape[0])
#     # print(img.shape[1])
#     # �任ͼ��ͨ��bgr->rgb
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#     #�ָ�
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

    # ��������
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # ����,����k=3;�к�ɫ�Ĳ���δ������ȡ
    compactness, labels, centers = cv2.kmeans(img_flat, 4, None, criteria, 10, flags)

    # ��ʾ���
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
    #תLab��HSL��ɫ�ռ�,������̦
    img_lab = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2LAB)
    img_hsl = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2HLS)
    cv2.imshow('shetai',img_shetai)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # plt.subplot(121), plt.imshow(img), plt.title('input')
    # plt.subplot(122), plt.imshow(img_shetai), plt.title('fin')
    # plt.show()
    #��̦����ֵ
    shetai_H,shetai_L1,shetai_S = cv2.split(img_hsl)
    exist = (shetai_L1 != 0)
    shetai_L_meanvalue = shetai_L1.sum()/exist.sum()

    #��̦a,bֵ
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
    #���ʷ���
    shezhi_R, shezhi_G, shezhi_B= cv2.split(img_shezhi)
    exist = (shezhi_R != 0)
    shezhi_R_meanvalue = shezhi_R.sum()/exist.sum()
    return shezhi_R_meanvalue

#ԭͼ

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
    #��ʴ������,���Ȼ�ø�ʴ�����Ͳ�����kernel
    kernel=np.zeros((kernel_size,kernel_size),np.uint8)
    for lenght in range(kernel_size_half):
        for x in range(kernel_size):
            for y in range(kernel_size):
                if (x-kernel_size_half)*(x-kernel_size_half)+(y-kernel_size_half)*(y-kernel_size_half)<=lenght*lenght:
                    kernel[x,y]+=int(255/kernel_size)+2

class Stats:

    def __init__(self):
        # ���ļ��м���UI����
        self.capIsOpen=0
        # �� UI �����ж�̬ ����һ����Ӧ�Ĵ��ڶ���
        # ע�⣺����Ŀؼ�����Ҳ��Ϊ���ڶ����������
        # ���� self.ui.button , self.ui.textEdit
        # self.ui = QUiLoader().load('./qt/ui/untitled.ui')

        # self.ui.button.clicked.connect(self.handleCalc)S

        self.mainPage=QUiLoader().load('./fea_all/ui/main.ui')#load��ҳ��
        # palette = QPalette()
        # # showimg('kkl',cv2.imread(".\qt\img\pizhi.jpg"))
        # k=QPixmap().load(".\qt\img\pizhi.jpg")
        # palette.setBrush(self.mainPage.backgroundRole(), QBrush(k)) #����ͼƬ
        # # palette.setBrush(QPalette.Background, QBrush(icon))
        # self.mainPage.setPalette(palette)
        self.display_video_stream(cv2.imread(".\\fea_all\img\logo.jpg"),self.mainPage.logo)
        self.mainPage.setWindowTitle("��ҽ��������ϵͳdemo-1.0")
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
        self.timer.start()            # ʵʱˢ�£���Ȼ��Ƶ����̬
        self.timer.setInterval(100)   # ����ˢ��ʱ��
        self.timer_4 = QTimer()
         # ʵʱˢ�£���Ȼ��Ƶ����̬
        self.bpm_list = []
        self.timer_4.setInterval(100)  # ����ˢ��ʱ��
        self.timer_4.start()
        self.questions=[["1�������Ƿ����ر�����������ȵ����",("A.��","B.��")],["2����û�г������ġ����ġ����з��ȵ����",("A.��","B.������֢״")],
       ["3�����ڵĳ���״����",("A.������֢״","B.˯��ʱ�׳���","C.�ճ������","D.�г��亹������")],
       ["4�������Ƿ��о�������ͷ�λ���ͷʹ��֢״",("A.��","B.��")],["5�������Ƿ������左����������ĸо���",("A.��","B.��")],
       ["6�������Ƿ��г������ᱳʹ�����������֢״",("A.��","B.��")],
       ["7�����ڴ���֢״",("A.������������̬(����)","B.��������̬�����ͣ�","C.�������������ճ��","D.�������������ϡ","E.����,��к")],
       ["8������θ��",("A.������","B.�Ե��٣���θ��","C.��ʳ��θ�ͣ�����","D.���������Ķ��������׶������Ż�¸�","���׶����Ҷ�������")],
       ["9�����ڵ�С���֢״",("A.��������������̬","B.С��̻�","C.С���峤","D.С������ζ")],
       ["10�����޳������ƣ��ļ£�в�ͺ͸��͵�֢״",("A.��","B.��")],
       ["11�����޳��ֶ����������½�������",("A.��","B.��")],
       ["12�����޾������ֿڸɻ��߿ڿʵ�֢״",("A.��","B.��")],
       ["13��ƽʱ����������������ǳ��£�",("A.����","B.����","C.����")],
       ["14�����޳���ʷ",("A.����","B.֬����","C.��Ѫѹ","D.��Ѫ֬","E.������","F.��Ѫѹ")]]
        self.nowQuestionIndex=0
        self.questionsLen=len(self.questions)
        self.ui_2.textBrowser.setText("�����ʼ����,���ı�����������,����ʵ�����������ѡ�����ѡ��һ����")
        self.answerTable=[]#��¼�ʾ�ѡ��Ĵ�1
        self.isAnswer=1
        self.singleAnswer=[]#��¼��������Ļش�,���Զ�ѡ
        self.shetou_fea_data=[]
        self.display_video_stream(cv2.imread(".\\fea_all\img\capBackground.jpg"),self.ui.label)

        self.stack=QStackedWidget(self.mainPage)
        self.stack.addWidget(self.ui)
        self.stack.addWidget(self.ui_2)
        self.stack.addWidget(self.ui_3)
        self.mainPage.playout.addWidget(self.stack)

        self.mainPage.listWidget.insertItem(0,'ͼ����ȡ����')
        self.mainPage.listWidget.insertItem(1,'�����ʴ𲿷�')
        self.mainPage.listWidget.insertItem(2,'���չʾ')
        self.mainPage.listWidget.currentRowChanged.connect(self.showPage)


    def shetou_fea(self):
        #ȡ�������°�
        img=self.img
        # img,flag = cut_face(img)
        # if flag == 0:
        #     QMessageBox.information(self.ui,"��ʾ","Ŀǰ������������,���ж�������ڼ��������,������")
        #     return
        #�����С
        img = img_resize(img)
        #���⻯
        equal = jun_hen(img)
        #������ֵ�˲�
        median = cv2.medianBlur(equal,5)
        #�и�����
        img = cut_she(median)


        self.tougueimg=img
        showimg("all",self.tougueimg)
######################################################


        #Kmeans
        centers,img_output = kmeans(img)
        print(centers)
        #showimg("P2",img_output)
        #�ҵ�λ
        shezhi_index,shetai_index = find(centers)
        #��̦����
        shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue = tai_fea(img_output,img,shetai_index)
        #���ʷ���
        shezhi_R_meanvalue = zhi_fea(img_output,img,shezhi_index)
        self.shetou_fea_data.extend([shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue,shezhi_R_meanvalue])
        self.ui.textEdit.setText(str(shetai_L_meanvalue))

    def xuanze(self):#����ģ��Ȩ��
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
        self.ui_2.toolButton_5.setText("��ʼ����")
        while self.ui_2.select_table.count():
            widget=self.ui_2.select_table.takeAt(0).widget()
            widget.deleteLater()
        self.ui_2.textBrowser.setText("�����ʼ����,���ı�����������,����ʵ�����������ѡ�����ѡ��һ����")

    def evaluate(self):
        self.sum_vector=np.zeros((9,))
        if self.face_vector[2]-self.face_vector[0]>6:
            self.sum_vector[0]+=1
            if self.face_vector.sum()>40:
                self.sum_vector[7]+=1
        if self.face_vector.sum()<20:
            self.sum_vector[7]+=1


        self.tongue_vector=[]

        self.status=["����","����","����","ƽ����","����","ʪ��","̵ʪ","Ѫ��","ƽ����"]

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
            QMessageBox.information(self.ui_2,"��ʾ","����ѡ���ٻش���һ��")
            return
        self.isAnswer=0#����Ϊδ�ش�
        if not self.singleAnswer ==[]:
            self.answerTable.append(self.singleAnswer)
        self.singleAnswer=[]
        i=self.nowQuestionIndex
        self.nowQuestionIndex+=1
        if i==0:
            self.ui_2.toolButton_5.setText("��һ��")
        if i==self.questionsLen-1:
            self.ui_2.toolButton_5.setText("��������")
        if i>self.questionsLen:
            QMessageBox.information(self.ui_2,"��ʾ","�Ѿ���������,��������Դ���,�밴���������")
            return
        if i==self.questionsLen:
            self.stack.setCurrentIndex(2)


            self.evaluate()
            self.ui_3.adviseinf.setText(str(self.answerTable))
            self.ui_3.bodyinf.setText(str(self.bodyinf))
            self.display_video_stream(self.faceimg,self.ui_3.faceimg)
            self.display_video_stream(self.tougueimg,self.ui_3.tougueimg)
            self.display_video_stream(self.rangeimg,self.ui_3.rangeimg)
            #QMessageBox.information(self.ui_2,"�����",str(self.answerTable)+"���ǳ�����!")
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
            self.ui.toolButton.setText("��Ƶ�ر�")
            self.cap = cv2.VideoCapture(0)
            self.timer.timeout.connect(self.capPicture)
        else:
            self.ui.toolButton.setText("��Ƶ����")
            self.hasimg=0
            self.display_video_stream(cv2.imread(".\\fea_all\img\capBackground.jpg"),self.ui.label)
            self.capIsOpen=0
            self.cap.release()
            self.ui.label.setText(" ")



    def faceFeature(self,test=1):#��ɫ��ȡ
        all_color_df=np.zeros((3))
        df_num=0

        for ll in range(10):
            if(self.cap.isOpened() or self.hasimg):
                start,second,three=35,6,3#�趨����,�ֱ��Ǹ�ʴ/���Ͳ�����kernel��С,��ʴ����,���ʹ���
                img = self.img #�õ���ǰ����Ƭ
                face_pos=self.face_catch.detectMultiScale(img,1.3,5)
                if(len(face_pos)>1 or len(face_pos)==0):
                  #  QMessageBox.information(self.ui,"��ʾ","Ŀǰ������������,���ж�������ڼ��������,������")
                    continue
                x,y,w,h=face_pos[0]
                img=img[y:y+h,x:x+w]#���������ͼƬ
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
                p1=cv2.erode(p1,kernel,iterations=second)#��ʴ
                #showimg("P1",p1)#35 6
                p1=cv2.dilate(p1,kernel,iterations=three)#����
                #showimg("P1",p1)#35 6
                cv2.normalize(p1,p1,0,255,norm_type=cv2.NORM_MINMAX)

                p13=p1
                p13=255-p13

                con,hie=cv2.findContours(p13,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                p3_copy=p13.copy()

                if len(con)==1:
                   # QMessageBox.information(self.ui,"��ʾ","Ŀǰ������������,���ж�������ڼ��������,������")
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
            self.ui.textEdit.setText(str([(k,i) for k,i in zip(['��','��','��'],self.face_vector)]))


### ������ȡ���ֿ�ʼ ###

    # ��ȡǰ�������
    def get_forehead_roi(self, face_points):
        # ����Щ��洢��Numpy�����У����Ժ����׵�ͨ����Ƭ�õ�x��y����Сֵ�����ֵ
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)
        # ǰ����ü������
        min_x = int(points[21, 0])
        min_y = int(min(points[21, 1], points[22, 1]))
        max_x = int(points[22, 0])
        max_y = int(max(points[21, 1], points[22, 1]))
        left = min_x
        right = max_x
        top = min_y - (max_x - min_x)
        bottom = max_y * 0.98
        return int(left), int(right), int(top), int(bottom)

    # ��ȡ���ӵ�����
    def get_nose_roi(self, face_points):
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)
        # ���Ӻ�����
        min_x = int(points[36, 0])
        min_y = int(points[28, 1])
        max_x = int(points[45, 0])
        max_y = int(points[33, 1])
        left = min_x
        right = max_x
        top = min_y + (min_y * 0.02)
        bottom = max_y + (max_y * 0.02)
        return int(left), int(right), int(top), int(bottom)

    # �����������������ɫֵȡƽ��ֵ
    def get_avg(self, roi1, roi2):
        roi1_green = roi1[:, :, 1]
        roi2_green = roi2[:, :, 1]
        avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
        return avg

    # ��ȡ����Ȥ�������ɫ����ƽ��ֵ��������Χ��һ����ɫ�ľ���
    def get_roi_avg(self, frame, face_points):  # ��Ƭ�������ؼ���
        # �õ�����Ȥ������
        fh_left, fh_right, fh_top, fh_bottom = self.get_forehead_roi(face_points)
        nose_left, nose_right, nose_top, nose_bottom = self.get_nose_roi(face_points)
        # ������Ȥ������(ROI)ƽ���ָ�
        fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
        nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
        return self.get_avg(fh_roi, nose_roi)

    # �������ڱ��
    def sliding_window_demean(self, signal_values, num_windows):
        window_size = int(round(len(signal_values) / num_windows))
        demeaned = np.zeros(signal_values.shape)
        for i in range(0, len(signal_values), window_size):
            if i + window_size > len(signal_values):
                window_size = len(signal_values) - i
            curr_slice = signal_values[i: i + window_size]
            demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
        return demeaned

    # ����ָ����Butterworth��������Ӧ��
    def butterworth_filter(self, data, low, high, sample_rate, order):
        nyquist_rate = sample_rate * 0.5
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)

    # �����ź�����
    def filter_signal_data(self, fps):  # ƽ����ɫ�������飬֡��
        # ȷ������û�����޻�NaNֵ
        values = np.array(self.roi_avg_values)
        np.nan_to_num(values, copy=False)
        # ͨ���������ƺͱ����ʹ�ź�ƽ��
        detrended = signal.detrend(values, type='linear')  # ���ź���ɾ����������
        demeaned = self.sliding_window_demean(detrended, 15)  # �������ڱ��
        # ��butterworth��ͨ�˲������źŽ����˲�
        filtered = self.butterworth_filter(demeaned, self.MIN_HZ, self.MAX_HZ, fps, 5)
        return filtered


    # ����ÿ���ӵ�����(BPM)
    def compute_bpm(self, filtered_values, fps, buffer_size):  # �˲������ݣ�֡������������С���ϴ�bpm
        # ���ٸ���Ҷ�任����FFT
        fft = np.abs(np.fft.rfft(filtered_values))
        # ������FFTֵ���Ӧ��Ƶ���б�
        fft_freq = fps / buffer_size * np.arange(buffer_size / 2 + 1)
        # �˳�FFT�в���[MIN_HZ, MAX_HZ]��Χ�ڵ�ȫ����ֵ����Ϊ��BPMֵ�ɶ�Ӧ����
        while True:
            max_idx = fft.argmax()
            bps = fft_freq[max_idx]
            if bps < self.MIN_HZ or bps > self.MAX_HZ:
                fft[max_idx] = 0
            else:
                bpm = bps * 60.0
                break
        # ������֮�䣬���ʱ仯�����ܳ���10%������ʹ�ü�Ȩƽ����ʹBPM�����һ��BPMƽ��
        if self.last_bpm > 0:
            bpm = (self.last_bpm * 0.9) + (bpm * 0.1)
        return bpm

    # ������ȡ��
    def pulse_observer(self):
        frame = self.img
        # ����޷�����������ͷ��ȡ ret_val == False
        if not self.ret_val:
            print("ERROR:  Unable to read from webcam.  Was the webcam disconnected?  Exiting.")
            return
        # �ڻ���֮ǰ�Ȱѻ�����һ�ݣ����ǽ���GUI����ʾ������ԭʼ֡�����ڼ�������
        view = np.array(frame)
        # ����ͼռ���ڿ�ȵ�75%��BPM���25%
        if self.graph_width == 0:
            self.graph_width = int(view.shape[1] * 0.75)
        if self.bpm_display_width == 0:
            self.bpm_display_width = view.shape[1] - self.graph_width
        # ʹ��dlib�������
        faces = self.detector(frame, 0)
        if len(faces) == 1:
            face_points = self.predictor(frame, faces[0])
            roi_avg = self.get_roi_avg(frame, face_points)
            self.roi_avg_values.append(roi_avg)
            self.times.append(time.time())
            # Buffer�Ѿ����ˣ��Ӷ�������ֵ��ɾ����
            if len(self.times) > self.BUFFER_MAX_SIZE:
                self.roi_avg_values.pop(0)
                self.times.pop(0)
            curr_buffer_size = len(self.times)
            # ������С֡��֮ǰ����Ҫ��������
            if curr_buffer_size > self.MIN_FRAMES:
                # ������صĴ���
                time_elapsed = self.times[-1] - self.times[0]
                fps = curr_buffer_size / time_elapsed  # ֡��
                # �����ź�����
                filtered = self.filter_signal_data(fps)
                self.graph_values.append(filtered[-1])
                if len(self.graph_values) > self.MAX_VALUES_TO_GRAPH:
                    self.graph_values.pop(0)
                # ���㲢��ʾBPM
                bpm = self.compute_bpm(filtered, fps, curr_buffer_size)
                self.bpm_list.append(bpm)
                if len(self.bpm_list) > self.BPM_MAX_SIZE:
                    self.bpm_list.pop(0)
                self.last_bpm = bpm
                self.ui.textEdit.setText(str(bpm))
            else:
                # ���û���㹻����������������������ʾһ�����м����ı���BPMռλ���Ŀ�ͼ
                pct = int(round(float(curr_buffer_size) / self.MIN_FRAMES * 100.0))
                self.ui.textEdit.setText('Computing pulse: ' + str(pct) + '%')

        else:
            # û�м�⵽�������Ա������ֵ��ʱ���б����򣬵��ٴμ�⵽����ʱ��ʱ��ͻ���ֿհס�
            del self.roi_avg_values[:]
            del self.times[:]

    def pulse_feature(self, event):
        self.BUFFER_MAX_SIZE = 500  # �洢�Ľ���ROIƽ��ֵ������
        self.MAX_VALUES_TO_GRAPH = 50  # ������ͼ����ʾ�����ROIƽ��ֵ
        self.BPM_MAX_SIZE = 10
        self.MIN_HZ = 0.83  # 50 BPM - ��С��������
        self.MAX_HZ = 3.33  # 200 BPM - �����������
        self.MIN_FRAMES = 100  # �ڼ�������֮ǰ�������С֡��
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
                self.ui.toolButton_4.setText("�ر����ʲɼ�")
                self.timer_Active = 1
                self.timer_4.timeout.connect(self.pulse_observer)

            else:
                self.timer_4.timeout.disconnect(self.pulse_observer)
                self.ui.toolButton_4.setText("���ʲɼ�")
                self.timer_Active = 0
        else:
            QMessageBox.information(self.ui,"��ʾ","����ͷδ����,������")
            return





### ������ȡ���ֽ��� ###


### ��ȡʵʱ����ͷ��Ƭ�Լ�չʾ ###
    def capPicture(self):
        if self.capIsOpen==1:#����ǿ�����Ƶ

            # get a frame
            ret, img = self.cap.read()#��ȡ����ͷ
            self.ret_val = ret
            self.img = img
            self.last=self.img#��һ�ε�ͼƬ
            height, width, bytesPerComponent = img.shape
            bytesPerLine = bytesPerComponent * width
            # �任��ɫ�ռ�˳��
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # תΪQImage����

            self.image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(self.image).scaled(self.ui.label.width(), self.ui.label.height()))








app = QApplication([])
stats = Stats()
stats.mainPage.show()
app.exec_()