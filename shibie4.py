# 导入所需库
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import time


# 获取人脸关键点
def get_landmarks(image, face_mesh):
    """
    :param image: ndarray图像
    :param face_mesh: 人脸检测模型
    :return:人脸关键点列表，如[{0:(x,y),1:{x,y},...},{0:(x,y),1:(x,y)}]
    """
    landmarks = []
    height, width = image.shape[0:2]
    # 人脸关键点检测
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 解释检测结果
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            i = 0
            points = {}
            # 根据图像的高度和宽度还原关键点位置
            for landmark in face_landmarks.landmark:
                x = math.floor(landmark.x * width)
                y = math.floor(landmark.y * height)
                points[i] = (x, y)
                i += 1
            landmarks.append(points)

    return landmarks


def process_effects(landmarks, icon_path, icon_name):
    """
    :param landmarks: 检测到的人脸关键点列表
    :param icon_path: 特效图像地址
    :param icon_name: 特效名称
    :return:处理好的特效图像、特效宽、特效高
    """
    # 特效关键点，用于调整特效的尺寸
    effect_landmarks = {"beard": ((landmarks[132][0], landmarks[5][1]), (landmarks[361][0], landmarks[0][1])),
                        "eyeglass": ((landmarks[127][0], landmarks[151][1]), (landmarks[356][0], landmarks[195][1])),
                        "halfmask": ((landmarks[162][0] - 50, landmarks[10][1] - 50),
                                     (landmarks[389][0] + 50, landmarks[195][1] + 50))}

    # 读取特效图像
    icon = cv2.imread(icon_path)
    print('读到特效图')
    # 选择特效关键点
    pt1, pt2 = effect_landmarks[icon_name]
    x, y, x_w, y_h = pt1[0], pt1[1], pt2[0], pt2[1]
    # 调整特效的尺寸
    w, h = x_w - x, y_h - y
    effect = cv2.resize(icon, (w, h))

    return effect, w, h


# 循环特效图像中每一个像素点，替换大于阈值的像素值
def swap_non_effcet1(effect,roi,threshold=240):
    """
    :param effect: 特效图像
    :param roi: ROI区域
    :param threshold: 阈值
    """
    for h in range(effect.shape[0]):
        for w in range(effect.shape[1]):
            for k in range(3):
                if effect[h][w][k] > threshold:
                    effect[h][w][k] = roi[h][w][k]

def swap_non_effcet2(effect, roi, threshold=240):
    """
    :param effect: 特效图像
    :param roi: ROI区域
    :param threshold: 阈值
    :return: 消除背景后的特效图像
    """

    # （1）特效图像灰度化
    effect2gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    # （2）特效图像二值化
    ret, effect2wb = cv2.threshold(effect2gray, threshold, 255, cv2.THRESH_BINARY)
    # （3）消除特效的白色背景
    effectwb = cv2.bitwise_and(roi, roi, mask=effect2wb)

    # （4）反转二值化后的特效
    effect2wb_ne = cv2.bitwise_not(effect2wb)
    # （5）处理彩色特效
    effectcolor = cv2.bitwise_and(effect, effect, mask=effect2wb_ne)
    # (6) 组合彩色特效与黑色特效
    effect_final = cv2.add(effectcolor, effectwb)

    return effect_final


# 模型配置
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=3,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
# 读取图像
image = cv2.imread("1.png")
print('读到原图')
# 获取关键点
face_landmarks = get_landmarks(image, face_mesh)
# ROI区域中心关键点
center = {"beard": 164, "eyeglass": 168, "halfmask": 9}

# 添加特效
image_copy = image.copy()
icon_name = "beard"

for landmarks in face_landmarks:
    # 调整特效尺寸
    effect, w, h = process_effects(landmarks, "icons/" + icon_name + ".png", icon_name)
    # 确定ROI
    p = center[icon_name]
    roi = image_copy[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
          landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w]
    if roi.shape[0:2] == effect.shape[0:2]:
        # 第一种方法
        #swap_non_effcet1(effect,roi,240)
        # 第二种方法
        swap_non_effcet2(effect, roi, 240)
        # 特效图像替换ROI区域
        image_copy[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
        landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w] = effect

cv2.imshow('texiao', image_copy)
key =  cv2.waitKey(1) & 0xFF


