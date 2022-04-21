import numpy as np
import cv2
def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    width_new = 540
    height_new = 540
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))

    return img_new
#鼠标事件的回调函数
def on_mouse(event,x,y,flag,param):
    global rect
    global leftButtonDowm
    global leftButtonUp

    #鼠标左键按下
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDowm = True
        leftButtonUp = False

    #移动鼠标事件
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDowm and  not leftButtonUp:
            rect[2] = x
            rect[3] = y

    #鼠标左键松开
    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDowm and  not leftButtonUp:
            x_min = min(rect[0],rect[2])
            y_min = min(rect[1],rect[3])

            x_max = max(rect[0],rect[2])
            y_max = max(rect[1],rect[3])

            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDowm = False
            leftButtonUp = True


img = cv2.imread(r'C:\Users\admin\Desktop\she_tou\sad.jpg')
img = img_resize(img)
mask = np.zeros(img.shape[:2],np.uint8)


bgdModel = np.zeros((1,65),np.float64)    #背景模型
fgdModel = np.zeros((1,65),np.float64)    #前景模型
rect = [0,0,0,0]                          #设定需要分割的图像范围


leftButtonDowm = False                    #鼠标左键按下
leftButtonUp = True                       #鼠标左键松开

cv2.namedWindow('img')                    #指定窗口名来创建窗口
cv2.setMouseCallback('img',on_mouse)      #设置鼠标事件回调函数 来获取鼠标输入
cv2.imshow('img',img)                     #显示图片


while cv2.waitKey(2) == -1:
    #左键按下，画矩阵
    if leftButtonDowm and not leftButtonUp:
        img_copy = img.copy()
        cv2.rectangle(img_copy,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)
        cv2.imshow('img',img_copy)

    #左键松开，矩形画好
    elif not leftButtonDowm and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
        rect[2] = rect[2]-rect[0]
        rect[3] = rect[3]-rect[1]
        rect_copy = tuple(rect.copy())
        rect = [0,0,0,0]
        #物体分割
        cv2.grabCut(img,mask,rect_copy,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img_show = img*mask2[:,:,np.newaxis]
        #显示图片分割后结果--显示原图
        x, y, w, h = rect_copy
        img_show = img_show[y:y+h,x:x+w]
        img_show = img_resize(img_show)
        cv2.imshow('grabcut',img_show)
        cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2


# src = cv2.imread(r'C:\Users\admin\Desktop\she_tou\sad.jpg')
# src = img_resize(src)
# proimage0 = src.copy()#复制原图


# roi = cv2.selectROI(windowName="roi", img=src, showCrosshair=False, fromCenter=False)#感兴趣区域ROI
# x, y, w, h = roi
# cv2.rectangle(img=src, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)#在图像绘制区域
# cv2.imshow("roi", src)

# #进行裁剪
# ImageROI=proimage0[y:y+h,x:x+w]
# ImageROI = img_resize(ImageROI)
# cv2.imshow("ImageROI", ImageROI)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
