import numpy as np
import cv2, os, shutil

unprocessed_path = "./unprocessed/"
face_path = "./input/face/"

def crop_image(img):
    face_detect = cv2.CascadeClassifier('haarcascade.xml')
    face_data = face_detect.detectMultiScale(img, 1.3, 5)

    cropped_img = None
    for (x, y, w, h) in face_data:
        x = x + 100
        y = y + 100

        cropped_img = img[y-290:y+h+100, x-200:x+w+100]

    # cv2.imwrite(f"./junk/test.png", cropped_img)
    return cropped_img

def blur_face(img):
    global count
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_roi = None
    img = crop_image(img)
    face_detect = cv2.CascadeClassifier('haarcascade.xml')
    face_data = face_detect.detectMultiScale(img, 1.3, 5)
    roi_count = 0
    for (x, y, w, h) in face_data: 
        roi_count = roi_count + 1
        x = x + 100
        y = y + 100
        roi = img[y:y+h-200, x:x+w-200]
        
        face_roi = img[y:y+roi.shape[0], x:x+roi.shape[1]]
        
        # applying a gaussian blur over this new rectangle area 
        roi = cv2.GaussianBlur(roi, (101, 101), 100)
        # impose this blurred image on original image to get final image 
        
        img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    cv2.imwrite("./junk/test.png", img)
    return img
    # cv2.imwrite(target_directory_path + filename, img)

for filename in os.listdir(unprocessed_path):

    img = cv2.imread(unprocessed_path + filename)
    
    # img = blur_face(img)
    img = crop_image(img)
    img = cv2.resize(img, (1024, 1024))
    cv2.imwrite(face_path+"new-"+filename.split(".")[0]+".png", img)
    # shutil.move(unprocessed_path+filename, face_path)