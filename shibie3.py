import mediapipe as mp
import cv2
import numpy as np

if __name__ == "__main__":

    # 构建脸部特征提取对象
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    # 构建绘图对象
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # # 开启摄像头
    # cap = cv2.VideoCapture()
    #
    # while True:
    #     # 读取一帧图像
    #     success, img = cap.read()
    #     if not success:
    #         continue
    #
    #     # 获取宽度和高低
    #     image_height, image_width, _ = np.shape(img)
    #
    #     # BGR 转 RGB
    #     img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     # 进行特征点提取
    #     results = face_mesh.process(img_RGB)
    #
    #     if results.multi_face_landmarks:
    #
    #         for face_landmarks in results.multi_face_landmarks:
    #
    #             # 自行计算478个关键点的坐标 并绘制
    #             if face_landmarks:
    #                 index_lip_up = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82,
    #                                 80, 191, 78, 61]
    #                 index_lip_down = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84,
    #                                   181, 91, 146, 61, 78]
    #                 pixel_size=5
    #                 # 计算关键点坐标
    #                 for i in index_lip_up:
    #                     pos_x = int(face_landmarks.landmark[i].x * image_width)
    #                     pos_y = int(face_landmarks.landmark[i].y * image_height)
    #
    #                     cv2.circle(img, (pos_x, pos_y), 10, (0, 255, 0), -1)
    #
    #     cv2.imshow("face-mesh", img)
    #     key = cv2.waitKey(1) & 0xFF
    #     # 按键 "q" 退出
    #     if key == ord('q'):
    #         break
    # cap.release()


img = cv2.imread('1.png')
# 获取宽度和高低
image_height, image_width, _ = np.shape(img)

# BGR 转 RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 进行特征点提取
results = face_mesh.process(img_RGB)

if results.multi_face_landmarks:

    for face_landmarks in results.multi_face_landmarks:

        # 自行计算478个关键点的坐标 并绘制
        if face_landmarks:
            index_lip_up = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82,
                            80, 191, 78, 61]
            index_lip_down = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84,
                              181, 91, 146, 61, 78]
            pixel_size=5
            # 计算关键点坐标
            for i in index_lip_up:
                pos_x = int(face_landmarks.landmark[i].x * image_width)
                pos_y = int(face_landmarks.landmark[i].y * image_height)

                cv2.circle(img, (pos_x, pos_y), 10, (0, 255, 0), -1)

    cv2.imshow("face-mesh", img)
    #key = cv2.waitKey(1) & 0xFF
#     # 按键 "q" 退出
#     if key == ord('q'):
#         break
# cap.release()