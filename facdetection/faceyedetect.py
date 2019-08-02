# Load opencv library
import cv2

# load face_cascade from haarcascade face xml file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# take image as input
img = cv2.imread("multiface.jpg")

# convert image into gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces 
faces = face_cascade.detectMultiScale(
    gray_img, scaleFactor=1.1, minNeighbors=10)

# create list of faces and make rectangle around faces
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

# resize image to fit 
resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

# show image
cv2.imshow("gray", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
