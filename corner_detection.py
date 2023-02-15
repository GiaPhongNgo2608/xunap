import cv2
import numpy as np
from IPython.display import Image

def detect_corner(image_path, blockSize=2, ksize=3, k=0.04, threshold=0.01):
    """
    image_path: link to image
    blockSize: the size of neighbourhood considered for corner detection
    ksize: parameter of Sobel derivative
    k: Harris detector free parameter in the equation.
    """
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>threshold*dst.max()]=[0, 0, 255] #  
    
    cv2.imshow('corner',img)
    cv2.waitKey()
    return img
if __name__ == "__main__":
  out_path = detect_corner('test2.jpg', blockSize=2, ksize=5, k=0.04, threshold=0.005)
  Image(out_path)