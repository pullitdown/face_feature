# %%
from pandas import cut
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from skimage import filters,io,color

# %%
def cv_show(name,img):
    cv2.imshow(name,img)

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    width_new = 1080
    height_new = 1080
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))

    return img_new

# %%
#temp = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(temp)

# %%
#分割部件

def cut_face(img):
    face_catch=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    face_pos=face_catch.detectMultiScale(img,1.3,5)
    x,y,w,h=face_pos[0]

    return img[int(y+0.8*h):int(y+1.2*h),int(x+0.3*w):int(x+0.7*w)]



# %%
#预处理

#均衡化----------------------------------------------------------------------------------------------------------
def jun_hen(img):
    B,G,R = cv2.split(img) #获得8通道的图像
    EB=cv2.equalizeHist(B)
    EG=cv2.equalizeHist(G)
    ER=cv2.equalizeHist(R)
    equal=cv2.merge((EB,EG,ER))
    return equal
    #plt.imshow(img)

# temp = cv2.cvtColor(equal,cv2.COLOR_BGR2RGB)
# plt.imshow(temp)

# %%
#消除噪声----------------------------------------------------------------------------------------------------------




# temp = cv2.cvtColor(median,cv2.COLOR_BGR2RGB)
# plt.imshow(temp)

# %%
#分割----------------------------------------------------------------------------------------------------------
def cut_she(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    SIZE = (1, 65)
    # print(img.shape[0])
    # print(img.shape[1])
    # 变换图像通道bgr->rgb
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #分割
    bgdModle = np.zeros(SIZE, np.float64)
    fgdModle = np.zeros(SIZE, np.float64)
    rect = (int((img.shape[0])/10), int((img.shape[1])/7), int((img.shape[0])*3/5), int((img.shape[1])*2/3))
    for i in range(3):
        cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 11, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img *= mask2[:, :, np.newaxis]
        cv2.grabCut(img, mask2, rect, bgdModle, fgdModle, 1)
        img *= mask2[:, :, np.newaxis]
    return img

# temp = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img)


# %%Kmeans
# 3个通道展平
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

# plt.subplot(121), plt.imshow(img), plt.title('input')
# plt.subplot(122), plt.imshow(img_output,'gray'), plt.title('kmeans')
# plt.show()
# centers

# %%
#查找舌苔和舌质的中心点位

def find(centers):
    centers_R = np.empty(shape = 4)
    ind = 0
    for i in centers:
        centers_R[ind] = i[0]
        ind = ind+1

    shezhi_index = np.argsort(centers_R)[-3]
    shetai_index = np.argsort(centers_R)[-1]
    return shezhi_index,shetai_index

# print(centers[shetai_index])
# print(centers[shezhi_index])

# %%
#使用mask获取舌苔部分

#舌苔分析
def tai_fea(img_output,img,shetai_index):
    mask4 = np.where(img_output==shetai_index,255,0)
    mask4 = np.uint8(mask4)
    img_shetai = cv2.bitwise_and(img, img, mask=mask4)
    #转Lab和HSL颜色空间,区分舌苔
    img_lab = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2LAB)
    img_hsl = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2HLS)
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

# %%
#使用mask获取舌质部分
def zhi_fea(img_output,img,shezhi_index):
    mask5 = np.where(img_output==shezhi_index,255,0)
    mask5 = np.uint8(mask5)
    img_shezhi = cv2.bitwise_and(img, img, mask=mask5)

    # plt.subplot(121), plt.imshow(img), plt.title('input')
    # plt.subplot(122), plt.imshow(img_shezhi), plt.title('fin')
    # plt.show()
    #舌质分析
    shezhi_R, shezhi_G, shezhi_B= cv2.split(img_shezhi)
    exist = (shezhi_R != 0)
    shezhi_R_meanvalue = shezhi_R.sum()/exist.sum()
    return shezhi_R_meanvalue

# %%
# img_lab = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2LAB)
# img_hsl = cv2.cvtColor(img_shetai,cv2.COLOR_RGB2HLS)

# %%
# hsv_list = cv2.calcHist([img_hsv],[0],None,[256],[0,256])
# hsv_list
# plt.hist(img_hsv.ravel(),255,[0,256])
# plt.show()

# channel = ('b','g','r')
# for i,col in enumerate(channel):
#     histr = cv2.calcHist([img_hsl],[0],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])

# plt.show()


# %%

# # %%
# img_gray = color.rgb2gray(img_shetai)#RGB转灰度
# frequency=0.6
# #调用gabor函数
# real, imag = filters.gabor(img_gray, frequency=0.6,theta=45,n_stds=5)
# #取模图像
# img_mod=np.sqrt(real.astype(float)**2+imag.astype(float)**2)
# #图像显示
# plt.figure()#总流程
#原图
img = cv2.imread('C:\\Users\\admin\\Desktop\\she_tou\\gs.jpg')
#取得脸部下半
img = cut_face(img)
#处理大小
img = img_resize(img)
#均衡化
equal = jun_hen(img)
#采用中值滤波
median = cv2.medianBlur(equal,5)
#切割舌体
img = cut_she(median)
#Kmeans
centers,img_output = kmeans(img)
#找点位
shezhi_index,shetai_index = find(centers)
#舌苔分析
shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue = tai_fea(img_output,img,shetai_index)
#舌质分析
shezhi_R_meanvalue = zhi_fea(img_output,img,shezhi_index)
# plt.subplot(2,2,1)
# plt.imshow(img_gray,cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(img_mod,cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(real,cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(imag,cmap='gray')
# plt.show()



print("shetai_L_meanvalue:{};\nshetai_A_meanvalue:{};\nshetai_B_meanvalue:{};\n".format(shetai_L_meanvalue,shetai_A_meanvalue,shetai_B_meanvalue))
print("shezhi_R_meanvalue:{};".format(shezhi_R_meanvalue))
print("finish")