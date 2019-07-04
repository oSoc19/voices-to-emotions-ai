import cv2
import numpy as np


def denoisify(file_path, dest_path):
    img = cv2.imread(file_path)
    print('Loaded: ' + file_path)
    blur = cv2.GaussianBlur(img, (15, 15), 2)
    rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    print(rgb)
    lower_bound = np.array([0, 35, 50])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(dest_path, masked_img)
