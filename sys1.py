#coding=gbk
from PySide2.QtWidgets import QApplication,QPushButton,QLabel,QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QBrush,QPixmap,QPalette,QImage
from PySide2.QtCore  import QTimer
import PySide2 
import cv2
import time
import numpy as np 
from PIL import Image
from numpy.__config__ import show
import dlib
from scipy import signal

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
        print(type(self.ui))
        self.ui.toolButton.clicked.connect(self.start)
        self.ui.toolButton_4.clicked.connect(self.pulse_feature)
        self.ui.toolButton_2.clicked.connect(self.faceFeature)

        self.timer_Active = 0
        self.timer = QTimer()
        self.timer.start()            # 实时刷新，不然视频不动态
        self.timer.setInterval(100)   # 设置刷新时间
        self.timer_4 = QTimer()
         # 实时刷新，不然视频不动态
        self.timer_4.setInterval(100)  # 设置刷新时间
        self.timer_4.start()
        


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
        

    def mouth_catch(self,event):#测试函数,无实际作用
        n=0.5
        start=eval(self.ui.lineEdit.text())
        second=eval(self.ui.lineEdit_2.text())
        three=eval(self.ui.lineEdit_3.text())
        if(self.cap.isOpened()):
            img = self.img #得到当前的照片
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            
            face_pos=self.face_catch.detectMultiScale(img,1.3,5)
            if(len(face_pos)>1):
                QMessageBox.information(self.ui,"提示","目前背景环境不佳,或有多个人脸在检测区域内,请重试")
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

    def faceFeature(self,test=0):#脸色提取
        if(self.cap.isOpened()):
            start,second,three=35,6,3#设定参数,分别是腐蚀/膨胀操作的kernel大小,腐蚀次数,膨胀次数
            img = self.img #得到当前的照片
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            
            face_pos=self.face_catch.detectMultiScale(img,1.3,5)
            if(len(face_pos)>1 or len(face_pos)==0):
                QMessageBox.information(self.ui,"提示","目前背景环境不佳,或有多个人脸在检测区域内,请重试")
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
            if test==1:
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
                QMessageBox.information(self.ui,"提示","目前背景环境不佳,或有多个人脸在检测区域内,请重试")
                return
            xm,ym,wm,hm= cv2.boundingRect(con[1]) 
             
            eye_pos=self.eye_catch.detectMultiScale(img)
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
            rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
            rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
            if test==1:
                showimg('face_rect', rect)
            color_df=np.zeros((3))
            for i,col in enumerate(['b','g','r']):
                hist_mask0=cv2.calcHist([img],[i],mask0,[25],[0,256])
                color_df[i]+=np.argmax(hist_mask0)
                hist_mask1=cv2.calcHist([img],[i],mask1,[25],[0,256])
                color_df[i]+=np.argmax(hist_mask1)
            color_df=color_df/2
            self.faceFeature=color_df
            self.ui.textEdit.setText(str([(k,i) for k,i in zip(['蓝','绿','蓝'],color_df)]))

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
                self.last_bpm = bpm
                self.ui.textEdit.setText(str(bpm))
            else:
                # 如果没有足够的数据来计算脉搏，则显示一个带有加载文本和BPM占位符的空图
                pass

        else:
            # 没有检测到脸，所以必须清除值和时间列表，否则，当再次检测到人脸时，时间就会出现空白。
            del self.roi_avg_values[:]
            del self.times[:]

    def pulse_feature(self, event):
        self.BUFFER_MAX_SIZE = 500  # 存储的近期ROI平均值的数量
        self.MAX_VALUES_TO_GRAPH = 50  # 在脉冲图中显示最近的ROI平均值
        self.MIN_HZ = 0.83  # 50 BPM - 最小允许心率
        self.MAX_HZ = 3.33  # 200 BPM - 最大允许心率
        self.MIN_FRAMES = 100  # 在计算心率之前所需的最小帧数
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./qt/data/s hape_predictor_68_face_landmarks.dat')
        self.roi_avg_values = []
        self.graph_values = []
        self.times = []
        self.last_bpm = 0
        self.graph_height = 200
        self.graph_width = 0
        self.bpm_display_width = 0

        if self.capIsOpen==1 :
            if self.timer_Active==0:
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
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            # 转为QImage对象

            self.image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(self.image).scaled(self.ui.label.width(), self.ui.label.height()))
                
        

            

        


app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()