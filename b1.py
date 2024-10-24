import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
img = cv2.imread('caycoi.jpg')

# Chuyển ảnh sang không gian màu grayscale (nếu cần)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Các hàm tăng cường ảnh
def negative(img):
    return 255 - img

def increase_contrast(img, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def log_transform(img, c=1):
    return c * np.log(1 + img)

def histogram_equalization(img):
    return cv2.equalizeHist(img)

# Áp dụng các phép biến đổi
img_negative = negative(gray)
img_contrast = increase_contrast(gray)
img_log = log_transform(gray)
img_histogram = histogram_equalization(gray)

# Hiển thị kết quả
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1), plt.imshow(gray, cmap='gray'), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(img_negative, cmap='gray'), plt.title('Negative')
plt.subplot(2, 3, 3), plt.imshow(img_contrast, cmap='gray'), plt.title('Increased Contrast')
plt.subplot(2, 3, 4), plt.imshow(img_log, cmap='gray'), plt.title('Log Transform')
plt.subplot(2, 3, 5), plt.imshow(img_histogram, cmap='gray'), plt.title('Histogram Equalization')
plt.show()