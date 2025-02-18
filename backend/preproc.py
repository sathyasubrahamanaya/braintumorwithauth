import cv2

def preproc(uploaded_image):
       
       image = cv2.imread(uploaded_image)
       image = cv2.resize(image, (64, 64))
       return image
