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
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sklearn
from sklearn.cluster import KMeans
from skimage import filters,io,color

##############################################################################################
#shetou
MAX_VALUES_TO_GRAPH = 50
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

def analyze(shezhi_R_meanvalue,shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue):
    #先判断舌苔
    tai = 0
    if shetai_L_meanvalue<=20 :
        tai = 3
        #print("灰黑苔")
    elif shetai_L_meanvalue>65:
        tai = 1
        #print("白苔")
    else:
        tai = 2
        #print("黄苔")

    #舌质判断

    start_val = 140
    kind = 5
    length = (255-start_val)/kind

    #1、偏白 2、淡红 3、偏红 4、暗红 5、老红
    if shezhi_R_meanvalue-start_val <0:
        shezhi_fin = 2
    else:
        shezhi_fin = int((shezhi_R_meanvalue-start_val)/length)
    #print(shezhi_fin)

    #status=["阴虚","阳虚","气虚","平和质","气郁","湿热","痰湿","血瘀","平和质"]

    fin = [0 for i in range(9)]
    if tai == 2:
        fin[5] = fin[5]+1
    elif tai == 1 and shezhi_fin==2:
        fin[2] = fin[2]+1
        fin[3] = fin[3]+1
    elif tai == 1 and shezhi_fin==1:
        fin[4] = fin[4]+1
    elif shezhi_fin==3:
        fin[0] = fin[0]+1
    elif shezhi_fin==4:
        fin[7] = fin[7]+1

    return fin

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
    root_path="./"
    def __init__(self):
        # 从文件中加载UI定义
        self.capIsOpen=0
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        # self.ui = QUiLoader().load('./qt/ui/untitled.ui')

        # self.ui.button.clicked.connect(self.handleCalc)S

        self.mainPage=QUiLoader().load(self.root_path+'ui/main.ui')#load主页面
        # palette = QPalette()
        # # showimg('kkl',cv2.imread(".\qt\img\pizhi.jpg"))
        # k=QPixmap().load(".\qt\img\pizhi.jpg")
        # palette.setBrush(self.mainPage.backgroundRole(), QBrush(k)) #背景图片
        # # palette.setBrush(QPalette.Background, QBrush(icon))
        # self.mainPage.setPalette(palette)
        self.display_video_stream(cv2.imread(self.root_path+'img\logo.jpg'),self.mainPage.logo)
        self.mainPage.setWindowTitle("中医养生建议系统demo-1.0")
        self.hasimg=0






        self.ui = QUiLoader().load(self.root_path+'ui/imgpage.ui')
        self.ui_2=QUiLoader().load(self.root_path+'ui/quespage.ui')
        self.ui_3=QUiLoader().load(self.root_path+'ui/goal.ui')
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
        self.questions=[
        ["1、近期是否有特别怕冷或者怕热的情况",("A.是","B.否")],
        ["2、有没有出现手心、脚心、胸中发热的情况",("A.有","B.无明显症状")],
        ["3、近期的出汗状况？",("A.无明显症状","B.睡觉时易出汗","C.日常生活汗多","D.有出冷汗的现象")],
        ["4、近期是否有经常出现头晕或者头痛的症状",("A.是","B.否")],
        ["5、近期是否有身体乏力，精神不振的感觉？",("A.是","B.否")],
        ["6、近期是否有出现腰酸背痛，精力不足的症状",("A.是","B.否")],
        ["7、近期大便的症状",("A.正常量且正常态(成型)","B.量少正常态（成型）","C.正常量，但大便粘滞","D.正常量，但大便稀","E.量多,腹泻")],
        ["8、近期胃口",("A.正常量","B.吃得少，无胃口","C.厌食，胃胀，泛酸","D.不想吃油腻的东西，容易对其产生呕吐感","容易饿，且多饮多尿")],
        ["9、近期的小便的症状",("A.正常量，且正常态","B.小便短黄","C.小便清长","D.小便有异味")],
        ["10、有无出现胸闷，心悸，胁胀和腹胀的症状",("A.有","B.无")],
        ["11、有无出现耳鸣和听力下降的问题",("A.有","B.无")],
        ["12、有无经常出现口干或者口渴的症状",("A.有","B.无")],
        ["13、平时多喝冷饮，热饮还是常温？",("A.冷饮","B.热饮","C.常温")],
        ["14、有无常病史",("A.糖尿病","B.脂肪肝","C.高血压","D.高血脂","E.高尿酸","F.低血压")]]


        self.advice = ["原则：食宜甘润，药宜养阴。避免过度劳累，忌熬夜，运动勿太过。\n起居养生：春天早睡早起；夏天晚睡早起，适当午睡；秋天早睡早起；冬天早睡晚起。炎热夏季注意避暑，避免剧烈运动。锻炼时要控制出汗量，及时补充水分。居室环境宜安静。应穿浅颜色散热透气性好的棉织、丝绸衣服。房事应有所节制。锻炼身体适合柔缓运动，不宜做汗出太多的剧烈运动，可打羽毛球、太极拳、散步等，以静为主，动静结合。\n饮食调养：多食甘凉滋润食物如绿豆、冬瓜、芝麻、百合、银耳、莲藕、木瓜、无花果、茼蒿、菠菜、雪梨、石榴、葡萄、柠檬、苹果、梨、香蕉、鸭肉、海参、蟹肉等。少食温燥辛辣香浓食物如羊肉、狗肉、葱、姜、蒜、辣椒、韭菜、葵花子、酒、咖啡等。银耳山药莲子粥、雪梨百合膏等。少吃油盐。\n",
                "原则：饮食宜温阳益气，起居要保暖，运动应避风寒。\n起居养生：春天早睡早起；夏天晚睡早起，适当午睡；秋天早睡早起；冬天早睡晚起。居住环境避寒就温，多晒太阳，多泡温泉，勤泡脚。夏天不宜剧烈运动，以免大汗亡阳。切忌连续熬夜损伤元气。房事应有所节制。冬天要选天气好的时间户外活动，避免寒冷损伤阳气。多运动，可散步、打太极拳升发阳气\n饮食调养：多食温补阳气及甘温益气食物如牛羊肉、狗肉、鹿肉、葱、姜、蒜、花椒、韭菜、辣椒、胡椒、桂圆、荔枝、腰果。少食生冷寒凉食物如黄瓜、藕、梨、西瓜、绿茶等及各种冷饮。夏勿贪凉，冬宜温补。药膳可选用当归生姜羊肉汤（当归15克、生姜五片、羊肉100克）、韭菜炒胡桃仁（胡桃仁20克、韭菜30克）。少吃油盐。\n",
                "原则：益气固本，健胃补脾。食宜甘温，药宜益气。\n起居养生：春天早睡早起；夏天晚睡早起，适当午睡；秋天早睡早起；冬天早睡晚起。起居规律，避免过度劳动和熬夜，免伤正气。运动要量力而行，坚持适度体育锻炼,可做一些柔缓的运动如散步、慢跑或打太极拳等；不宜做大负荷或出大出汗的运动,避免汗出伤风。\n饮食调养：宜吃性平偏温，具有益气健脾作用的食物如糯米、小米、黄豆、白扁豆、红薯、土豆、山药、胡萝卜、大枣、桂圆、苹果、龙眼肉、莲子、板栗、牛羊肉、鸡肉、鲢鱼、香菇、蜂蜜等。少吃青萝卜、槟榔等耗气食物。饮茶宜选红茶，不宜饮绿茶。药膳可选用茯苓粳米粥（茯苓12克，粳米100克）、山药桂圆粥（山药100克，桂圆15克）及粳米山药莲子粥、黄芪母鸡汤、人参汤。少吃油、盐。\n",
                "自由项\n",
                "原则：药宜理气，食宜辛温宽胸，起居宜动不宜静，多参加群体运动。\n起居养生：春天早睡早起，夏天晚睡早起、适当午睡，秋天早睡早起，冬天早睡晚起。多进行户外活动，不要总待在家，要放松身心，和畅气血。居住要安静，防止嘈杂影响心情。睡前要避免饮茶、喝咖啡等提神醒脑的饮料\n饮食调养：多食宽胸理气食物如小麦、黄花菜、白萝卜、海带、海藻、大葱、洋葱，蒜、开心果、茴香、佛手、柚子、橙子、柑子、刀豆、金橘等。少吃收敛酸涩食品，如酸菜、乌梅、石榴、青梅、杨梅、酸枣、李子、柠檬、杨桃及寒凉滋腻食品。宜饮玫瑰茉莉花茶，少量饮酒。药膳可用橘皮粳米粥（橘皮50g、粳米100g）、山药佛手冬瓜汤（山药50克、佛手50克、冬瓜150克）。少吃油盐。\n",
                "原则：食宜清淡忌辛温，药宜清热祛湿，居避暑湿，增强运动。\n起居养生：春天早睡早起；夏天晚睡早起，适当午睡；秋天早睡早起；冬天早睡晚起。戒除烟酒，不要熬夜。多户外活动，常晒太阳。暑湿季节减少户外活动，避免淋雨受寒及感受暑湿，保持居室干燥。衣着宽松，面料以棉麻丝等天然纤维为主。保持二便通畅。穿衣宽松透气。适合强度锻炼如长跑、爬山、游泳、打球以消耗体内多余热量。\n饮食调养：多食清淡食品如空心菜、苋菜、芹菜、荠菜、苦瓜、黄瓜、冬瓜、藕节等蔬菜。适量吃西瓜、绿豆、赤小豆、豆腐、薏苡仁、鸭肉。少食辛温助热食物如韭菜、生姜、花椒、辣椒、麻辣油炸食物。忌食辛温滋腻食品如羊肉、狗肉或火锅、烹炸、烧烤以及肥肉、粘糕、鲇鱼等粘腻食物。少吃油盐少饮酒，多喝白开水及凉茶。可用石竹茶、苦丁茶、莲子芯、竹叶、玉米须等泡茶饮。\n",
                "原则：化痰祛湿，药宜温化，饮食宜清淡，起居恶潮湿，运动宜渐进。\n起居养生：春天早睡早起；夏天晚睡早起，适当午睡；秋天早睡早起；冬天早睡晚起。天气晴朗时多进行户外活动，常晒太阳或进行日光浴。气候阴冷时减少户外活动，避免受寒淋雨。居室宜温暖干燥，不宜阴冷潮湿。坚持锻炼如散步、慢跑，运动量力而行，循序渐进，以微微汗出为佳。不宜过于安逸，贪睡卧床。衣着宽松，面料以棉麻丝等天然纤维为主。\n",
                "原则：食宜辛温，药宜活血，加强运动促使血液运行。\n起居养生：春天早睡早起，夏天晚睡早起、适当午睡，秋天早睡早起，冬天早睡晚起。居室宜温不宜凉。不可过于安逸，少用电脑少熬夜，以免气机郁滞而致血行不畅。衣着宽松。多走出户外活动，舒展形体，放松心情。\n饮食调养：宜食黑豆、海带、紫菜、萝卜、胡萝卜、醋、绿茶、山楂、金橘、玫瑰花、桃花、月季花、田七苗、油菜、桃仁等具有活血散结行气及疏肝解郁的食物。忌吃乌梅、苦瓜、柿子、李子、石榴等酸涩之品及寒凉食物。少吃蛋黄、蟹子、猪肉、奶酪等滋腻之品。适量饮用葡萄酒、黄酒等。药膳可选用山楂汤（山楂20克加少许红糖）、当归田七乌鸡汤（乌鸡250克、当归15克、田七10克、生姜五片）或黑豆川芎粥、红花煎等。少吃油盐。\n",
                "原则：重在维护，饮食有节，劳逸结合，坚持锻炼。\n起居养生：生活起居顺应四季气候特点，保证充足睡眠。春天早睡早起；夏天晚睡早起，适当午睡；秋天早睡早起；冬天早睡晚起。根据气候变化适时增减衣物，不可过度劳累，不伤不扰，顺其自然。\n饮食调养：常吃五谷杂粮、瓜果蔬菜；少食油腻及辛辣食品，戒烟限酒。进食应有所节制，不可过饥过饱。不要偏寒偏热，少吃油盐。\n"
                ]
        self.nowQuestionIndex=0
        self.questionsLen=len(self.questions)
        self.ui_2.textBrowser.setText("点击开始答题,本文本框会出现问题,根据实际情况在以下选择框内选择一个答案")
        self.answerTable=[]#记录问卷选择的答案1
        self.isAnswer=1
        self.singleAnswer=[]#记录单个问题的回答,可以多选
        self.shetou_fea_data=[]
        self.display_video_stream(cv2.imread(self.root_path+'img\capBackground.jpg'),self.ui.label)

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
        self.tongue_vector = analyze(shezhi_R_meanvalue,shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue)
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
        self.sum_vector = self.sum_vector+np.array(self.tongue_vector)



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
        self.adviceinf=""
        for i in sta:
            self.bodyinf.append(self.status[i])
            self.adviceinf+=self.advice[i]




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

            plt.plot(range(len(self.graph_values)), self.graph_values)
            # 将plt转化为numpy数据
            canvas = FigureCanvasAgg(plt.gcf())
            # 绘制图像
            canvas.draw()
            # 获取图像尺寸
            w, h = canvas.get_width_height()
            # 解码string 得到argb图像
            buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
            # 重构成w h 4(argb)图像
            buf.shape = (w, h, 4)
            # 转换为 RGBA
            buf = np.roll(buf, 3, axis=2)
            # 得到 Image RGBA图像对象 
            image = Image.frombytes("RGBA", (w, h), buf.tostring())
            # 转换为numpy array rgba四通道数组
            image = np.asarray(image).astype(np.uint8)
            # 转换为rgb图像
            self.rangeimg = image[:, :, :3]
            

            self.ui_3.adviseinf.setText(self.adviceinf)
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
            self.display_video_stream(cv2.imread(self.root_path+'img\capBackground.jpg'),self.ui.label)
            self.capIsOpen=0
            self.cap.release()
            self.ui.label.setText(" ")



    def faceFeature(self,test=1):#脸色提取
        all_color_df=np.zeros((3))
        df_num=0
        if(self.cap.isOpened() or self.hasimg):

            for ll in range(10):
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
    
    # 返回列表中的最大绝对值
    def get_max_abs(lst):
        return max(max(lst), -min(lst))

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
        graph_height = 200
        graph_width = 0
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
                #绘制心率图

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
                loading_text = 'Computing pulse: ' + str(pct) + '%'

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
        self.predictor = dlib.shape_predictor(self.root_path+'data/shape_predictor_68_face_landmarks.dat')
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