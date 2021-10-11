#coding=gbk
from PySide2.QtWidgets import QApplication,QPushButton,QLabel,QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QBrush,QPixmap,QPalette,QImage
from PySide2.QtCore  import QTimer
import PySide2 
import cv2
import numpy as np 
from PIL import Image
from numpy.__config__ import show

def showimg(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows() 

def showbignum(name,img):
    print(name,"\n:\n",img)
    cv2.normalize(img,img,0,255,norm_type=cv2.NORM_MINMAX) 
    img = cv2.convertScaleAbs(img)
    print(name,"\n:\n",img)

    showimg(name,img)
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

        # self.ui.button.clicked.connect(self.handleCalc)
        # palette = QPalette()
        # icon = QPixmap(r'C:\Users\stepf\Desktop\pizhi.jpeg').scaled(800, 600)
        # palette.setBrush(self.backgroundRole(), QBrush(icon))
        # self.setPalette(palette)
        self.face_catch=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

        self.eye_catch=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
        self.ui = QUiLoader().load('./qt/ui/untitled.ui')

        self.ui.toolButton.clicked.connect(self.start)
        self.ui.toolButton_2.clicked.connect(self.faceFeature)
        #self.ui.toolButton_2.clicked.connect(self.mouth_catch)
        
        
        self.timer = QTimer()
        self.timer.start()            # 实时刷新，不然视频不动态
        self.timer.setInterval(100)   # 设置刷新时间


    def start(self,event):
        if self.capIsOpen==0:
            self.capIsOpen=1
            self.ui.toolButton.setText("视频关闭")
            self.cap = cv2.VideoCapture(0)
            self.timer.timeout.connect(self.capPicture)
        else:
            self.ui.toolButton.setText("视频开启")
            self.capIsOpen=0
            self.cap.release()
            self.ui.label.setText(" ") 
        

    def mouth_catch(self,event):
        n=0.5
        start=eval(self.ui.lineEdit.text())
        second=eval(self.ui.lineEdit_2.text())
        three=eval(self.ui.lineEdit_3.text())
        if(self.cap.isOpened()):
            img = self.img #得到当前的照片
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            
            face_pos=self.face_catch.detectMultiScale(img,1.3,5)
            if(len(face_pos)>1):
                QMessageBox.information("目前背景环境不佳,或有多个人脸在检测区域内,请重试")
                return
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
            
            # showimg("CR", CR)
            # showimg("CB", CB)
            # showimg("CR_arr",CR_array)
            # showimg("CB_arr",CB_array)
            # showimg("CR2",CR2)
            # showimg("CRB",CRB)
            #showimg("P1",p1)
            kernel=kernelbysize(int(np.floor(img.shape[0]/5)))                
            p1=cv2.erode(p1,kernel,iterations=second)#腐蚀
            #showimg("P1",p1)#35 6
            p1=cv2.dilate(p1,kernel,iterations=three)#膨胀
            #showimg("P1",p1)#35 6 
            cv2.normalize(p1,p1,0,255,norm_type=cv2.NORM_MINMAX)
            showimg("P1",p1)#35 6 3
            con,hie=cv2.findContours(p1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            p1_copy=p1.copy()
            print(len(con))
            for i in range(len(con)):
                con[1]

    def faceFeature(self,event):
        if(self.cap.isOpened()):
            start,second,three=35,6,3
            img = self.img #得到当前的照片
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            
            face_pos=self.face_catch.detectMultiScale(img,1.3,5)
            if(len(face_pos)>1):
                QMessageBox.information("目前背景环境不佳,或有多个人脸在检测区域内,请重试")
                return
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
            
            showimg("CR", CR)
            showimg("CB", CB)
            # showimg("CR_arr",CR_array)
            # showimg("CB_arr",CB_array)
            # showimg("CR2",CR2)
            # showimg("CRB",CRB)
            #showimg("P1",p1)
            kernel=kernelbysize(int(np.floor(img.shape[0]/5)))                
            p1=cv2.erode(p1,kernel,iterations=second)#腐蚀
            #showimg("P1",p1)#35 6
            p1=cv2.dilate(p1,kernel,iterations=three)#膨胀
            #showimg("P1",p1)#35 6 
            cv2.normalize(p1,p1,0,255,norm_type=cv2.NORM_MINMAX)
            showimg("P1",p1)#35 6 3
            p13=p1
            p13=255-p13
            showimg("p13",p13)
            con,hie=cv2.findContours(p13,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            p3_copy=p13.copy()
            print(len(con)) 
            xm,ym,wm,hm= cv2.boundingRect(con[1]) 
             
            eye_pos=self.eye_catch.detectMultiScale(img)
            x0,y0,w0,h0=eye_pos[0]
            x1,y1,w1,h1=eye_pos[1]
            fin=img.copy()
            rect=cv2.rectangle(fin,(xm,ym),(xm+wm,ym+hm),(0,255,0),3)
            rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
            rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
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
            showimg('rect', rect)
            mask0=np.zeros(img.shape[:2],np.uint8)

            mask0[y0:y0+h0,x0:x0+w0]=255
            mask1=np.zeros(img.shape[:2],np.uint8)
            mask1[y1:y1+h1,x1:x1+w1]=255
            fin=img.copy()
            rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
            rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
            showimg('rect', rect)
            color_df=np.zeros((3))
            for i,col in enumerate(['b','g','r']):
                hist_mask0=cv2.calcHist([img],[i],mask0,[25],[0,256])
                color_df[i]+=np.argmax(hist_mask0)
                hist_mask1=cv2.calcHist([img],[i],mask1,[25],[0,256])
                color_df[i]+=np.argmax(hist_mask1)
            color_df=color_df/2
            self.faceFeature=color_df
            self.ui.textEdit.setText(str([(k,i) for k,i in zip(['蓝','绿','蓝'],color_df)]))
    
    def capPicture(self):
        if self.capIsOpen==1:
            
            # get a frame
            ret, img = self.cap.read()
            self.img=img
            self.last=self.img
            height, width, bytesPerComponent = img.shape
            bytesPerLine = bytesPerComponent * width
            # 变换彩色空间顺序
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            # 转为QImage对象

            self.image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(self.image).scaled(self.ui.label.width(), self.ui.label.height()))
                
        

            

        


app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()