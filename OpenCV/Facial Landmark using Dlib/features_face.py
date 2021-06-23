import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print (predictor)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()


        landmarks = predictor(gray, face)


        x = landmarks.part(20).x
        y = landmarks.part(20).y

        g = landmarks.part(23).x
        q = landmarks.part(23).y
        print(x1,y1 , x2,y2)
        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        cv2.circle(frame, (g, q), 4, (255, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (x, y-40), (g, q), (0, 255, 0), 1)
        cv2.putText(frame, 'Forehead Detected' , (g-80,q+50), font , 1, (255,255,255) ,1, cv2.LINE_AA )

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break