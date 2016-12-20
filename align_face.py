import cv2
import os
import pickle
import time
import copy

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
import random

dlib_face_predictor_path = "/home/timpfey/Work/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
# Sherbakov_test2.png
img_path = '/home/timpfey/Work/openface/test-images/Kataev_test1.png'
opencv_cascade_path = '/home/timpfey/Work/opencv/data/haarcascades/'

align = openface.AlignDlib(dlib_face_predictor_path)
face_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'Nariz.xml')
mouth_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'Mouth.xml')


def dlib_align(img):
    # get reps
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get points
    start_time = time.time()
    # detect face
    bb = align.getAllFaceBoundingBoxes(rgb_img)
    print ("detect face {}".format(time.time() - start_time))
    # get aligned faces
    aligned_faces = []
    for box in bb:
        start_time = time.time()
        aligned_faces.append(
            align.align(
                96,
                rgb_img,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
        print ("align face {}".format(time.time() - start_time))
    return aligned_faces


def opencv_search_senses(gray_img, scale_factor=1.1, min_neighbors=3):

    eye = eye_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)
    #nose = nose_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)
    mouth = mouth_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)

    return {'eye': eye, 'nose': nose, 'mouth': mouth}

def opencv_find_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detect = {}
    for (x, y, w, h) in faces:
        detect = {'face':[x, y, w, h]}
        roi_gray = gray[y:y + h, x:x + w]
        sen = opencv_search_senses(roi_gray, 1.1, 2)
    detect.update(sen)

    return detect

if __name__ == '__main__':
    random.seed(time.time())

    img = cv2.imread(img_path)

    aligned_faces = dlib_align(img)

    i = 0
    for face in aligned_faces:
        cv2.imshow("align face{}".format(i), face)
        cv2.imwrite("/home/timpfey/Work/openface/test-images/test-align/dlib_align_test{}.png".format(i), face)
        i += 1

    gray = cv2.cvtColor(aligned_faces[0], cv2.COLOR_BGR2GRAY)
    senses = opencv_search_senses(gray)

    img_copy = copy.copy(aligned_faces[0])
    for name, rect in senses.iteritems():
        for (x, y, w, h) in rect:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_copy, '{}'.format(name), (x+w/2, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow('opencv align senses', img_copy)

    cv2.imshow('origin', img)
    detect = opencv_find_face(img)
    fx, fy, fw, fh = detect['face']
    origin_copy = copy.copy(img)
    face_origin = origin_copy[fy:fy + fh, fx:fx + fw]
    for name, rect in detect.iteritems():
        if name != 'face':
            for (x, y, w, h) in rect:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(face_origin, (x, y), (x + w, y + h), color, 2)
                cv2.putText(face_origin, '{}'.format(name), (x + w / 2, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('opencv face', face_origin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()