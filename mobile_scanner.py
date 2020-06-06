#detect edges
#use edges to find contour of the page
#perspective transform to align to the contour
''' from birds_perspective_rectangular_warp import four_point_transform
--->This one does the same as the library function with same name in imutils
   *Refer birds_perspective_rectangular_warp.py '''
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils


img=cv2.imread("page.jpg")
ratio=img.shape[0]/500
#original image
orig=img.copy()
img=imutils.resize(img,height=500)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Gaussian blur applied to detect the corners properly,then in turn detect edges
gray=cv2.GaussianBlur(gray,(5,5),0)
edge_detected=cv2.Canny(gray,75,200)
cv2.imshow("Edged",edge_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()
cont=cv2.findContours(edge_detected.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cont=imutils.grab_contours(cont)
cont=sorted(cont,reverse=True,key=cv2.contourArea)[:5]# to store only the largest contours.
for c in cont:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            pt=approx
            break

cv2.drawContours(img,[pt],-1,(145,71,145),2)
cv2.imshow("Contoured",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
warped = four_point_transform(orig, pt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
