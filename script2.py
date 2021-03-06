import cv2
import glob

# loop over folder with specific extention
images = glob.glob('*.jpg')

for image in images:
    print(image)
    img = cv2.imread(image,0)
    re = cv2.resize(img,(100,100))
    cv2.imshow("Image",re)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.imwrite("resized_"+image,re)