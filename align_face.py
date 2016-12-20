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
face_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'face.xml')
eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'Eye.xml')
nose_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'nose.xml')
mouth_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'mouth.xml')
right_eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'right_eye.xml')
left_eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'left_eye.xml')


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
    eye = eye_cascade.detectMultiScale(gray_img)
    nose = nose_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)
    if len(nose) > 0 and len(eye) > 0:
        return {'eye': eye[0], 'nose': [nose[0]], 'mouth': []}
    else:
        return None


def opencv_search_eyes(gray_img, scale_factor=1.1, min_neighbors=3):
    rigth_eye = right_eye_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)
    left_eye = left_eye_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)
    return {'r_eye': rigth_eye, 'l_eye': left_eye}


def opencv_find_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    x, y, w, h = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    color_face = img[y:y + h, x:x + w]
    roi_gray96 = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_CUBIC)
    color_face96 = cv2.resize(color_face, (96, 96), interpolation=cv2.INTER_CUBIC)
    return roi_gray96, color_face96


def get_nose_point(nose_rect):
    x = nose_rect[0]
    y = nose_rect[1]
    w = nose_rect[2]
    h = nose_rect[3]
    return [x+w/2, y+h/2]


def get_eye_points(eye_rect):
    lx = eye_rect[0]
    ly = eye_rect[1]
    w = eye_rect[2]/2
    h = eye_rect[3]/2
    rx = lx + w
    ry = ly + h

    return [[lx + w/2, ly + h/2], [rx + w/2, ry + h/2]]


def draw_rect(rect_dist, img):
    for name, rect in rect_dist.iteritems():
        for (x, y, w, h) in rect:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, '{}'.format(name), (x+w/2, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_points(points, img):
    for point in points:
        cv2.circle(img, (point[0], point[1]), 1, (255, 255, 0))
        cv2.circle(img, (point[0][0], point[0][1]), 1, (0, 255, 255))
        cv2.circle(img, (point[1][0], point[1][1]), 1, (0, 255, 255))


if __name__ == '__main__':

    random.seed(time.time())

    img = cv2.imread(img_path)
    aligned_faces = dlib_align(img)

    gray = cv2.cvtColor(aligned_faces[0], cv2.COLOR_BGR2GRAY)
    senses = opencv_search_senses(gray)

    dlib_nose_point = get_nose_point(senses['nose'][0])
    dlib_eye_point = get_eye_points(senses['eye'])


    gray_face, opencv_face = opencv_find_face(img)
    cv_face = copy.copy(opencv_face)
    cv_senses = opencv_search_senses(gray_face)

    nose_point = get_nose_point(cv_senses['nose'][0])
    eye_points = get_eye_points(cv_senses['eye'])

    cv2.waitKey(0)
    cv2.destroyAllWindows()