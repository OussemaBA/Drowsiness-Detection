#!/usr/bin/python
from scipy.spatial import distance
import dlib
import cv2
from imutils import face_utils
from threading import Thread
import argparse
import imutils
from playsound import playsound
import sys

def sound_alarm(path):
    # play an alarm sound
    playsound(path)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
args = vars(ap.parse_args())

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture('demo.mp4')
#cap = cv2.VideoCapture(0)
flag = 0
ALARM_ON = False
if args["alarm"] != "":
    t = Thread(target=sound_alarm,
    args=(args["alarm"],))
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "Reveillez-vous!", (150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Reveillez-vous!", (150, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not ALARM_ON:
                    ALARM_ON = True
                     #vérifier si un fichier d'alarme a été fourni,
                    # et si c'est le cas, démarrez un thread pour que le son d'alarme soit joué
                    # en arrière-plan
                    t.deamon = True
                    t.start()
        else:
            flag = 0
            ALARM_ON = False
            t.deamon = True
 
           

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()
