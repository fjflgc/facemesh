import cv2

# 加载原始图像
image = cv2.imread('1.png')

# 设置马赛克区域的坐标和大小
x, y, w, h = 100, 100, 200, 200

# 获取马赛克区域
roi = image[y:y+h, x:x+w]

# 将马赛克区域缩小
roi_small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
roi_back = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)

# 将缩小后的区域放回原图像
image[y:y+h, x:x+w] = roi_back

# 保存效果图像
cv2.imwrite('output_image.jpg', image)

# 显示效果图像
cv2.imshow('Mosaic Effect', image)
cv2.waitKey(0)
cv2.destroyAllWindows()