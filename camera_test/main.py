from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2
import numpy as np
import serial
import dlib



    
class Face_detector:
    def __init__(self, scale_factor, minNeighbors,cascade_path = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'):
        print('Cascade_path:', cascade_path)
        self.face_cascade= cv2.CascadeClassifier(cascade_path)
        self.scale_factor= scale_factor
        self.min_neighbors = minNeighbors
    def __call__(self,image):
        if len(image.shape) >2:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        else:
            gray = np.copy(image)


        faces= self.face_cascade.detectMultiScale(gray,self.scale_factor,self.min_neighbors)
        return faces
    def draw_faces(self, image, faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        return image
    def get_face_centers(self,faces):
        centers = []
        for  (x,y,w,h) in faces:
            centers.append(((x+w/2),(y+h/2)))
        return centers


tracker = dlib.correlation_tracker()

tracking_face =0
#camera = PiCamera()
cv2.namedWindow('image',cv2.WINDOW_NORMAL)




with PiCamera() as camera:
    camera.resolution = (640,480)
    camera.framerate=10
    rawCapture = PiRGBArray(camera, size=(640,480))
    key = 0
    face_detector = Face_detector(1.1,5)
    #,cascade_path = cv2.data.haarcascades+'lbpcascade_frontalface_improved.xml')
    
    time.sleep(2)
    i = 0
    for frame in camera.capture_continuous(rawCapture,format='bgr', use_video_port=True):
        image = frame.array
        cv2.imshow('image',image)

        if not tracking_face:
            faces = face_detector(image)
            maxArea = 0
            x =0
            y = 0
            h = 0
            w = 0
            for(_x,_y,_w,_h) in faces:
                if _w*_h > maxArea:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)
                    maxArea = w*h
            if maxArea >0:
                tracker.start_track(image,dlib.rectangle(x-10,y-20,x+w+10, y+h+20))
                tracking_face = 1
        if  tracking_face:
            trackingQuality = tracker.update(image)
            if trackingQuality >= 8.75:
                tracking_position = tracker.get_position()
                t_x = int(tracking_position.left())
                t_y = int(tracking_position.top())

                t_w = int(tracking_position.width())
                t_h = int(tracking_position.height())
                faces = [(t_x,t_y,t_w,t_h)]
            else:
                tracking_face = 0
        results = face_detector.draw_faces(image,faces)
        cv2.imshow('Results',results)

        key = cv2.waitKey(90) & 0xFF
        print('Image {} : size {}'.format(i,rawCapture.tell()))
        rawCapture.seek(0)
        i= i+1
        rawCapture.truncate(0)
        if key ==ord("q") or i ==1200:
            break
    cv2.destroyAllWindows()
    
