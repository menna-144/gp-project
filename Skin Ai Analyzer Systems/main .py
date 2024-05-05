import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skimage import filters

def calculate_redness_score(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return 0, None

    x, y, w, h = faces[0]
    face_roi = img[y:y+h, x:x+w]
    face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(face_hsv, lower_red, upper_red)
    redness_percentage = (np.count_nonzero(mask) / (w * h)) * 100

    edge_mask = filters.sobel(mask)
    segmented_image = cv2.watershed(face_roi, edge_mask)

    regressor = RandomForestRegressor()
    X = np.array([[x, y, w, h] for x, y, w, h in faces])
    y = np.array([redness_percentage for _ in range(len(faces))])
    regressor.fit(X, y)

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(cv2.bitwise_and(face_roi, face_roi, mask=mask), cv2.COLOR_BGR2RGB))
    plt.title('Red Mask')

    plt.subplot(2, 2, 3)
    plt.imshow(edge_mask, cmap='gray')
    plt.title('Edge Mask')

    plt.subplot(2, 2, 4)
    plt.imshow(segmented_image)
    plt.title('Segmented Image')

    plt.show()

    return redness_percentage, mask

image_path = 'la-roche-posay-what-causes-adult-acne-and-how-to-treat-it.webp'

redness_score, red_mask = calculate_redness_score(image_path)
print(f'Redness Score: {redness_score:.2f}')

if red_mask is not None:
    cv2.imwrite('red_mask.jpg', red_mask)
