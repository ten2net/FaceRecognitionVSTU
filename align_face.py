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
net_path_128 = "/home/timpfey/Work/openface/models/openface/nn4.small2.v1.t7"
svm_path = "/home/timpfey/Work/openface/generated-embeddings/classifier.pkl"
im_path = "/home/timpfey/Work/openface/test-images/Screenshot_20161213_014837.png"
opencv_cascade_path = '/home/timpfey/Work/opencv/data/haarcascades/'
img_path = '/home/timpfey/Work/openface/test-images/Kataev_test1.png'

face_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'face.xml')
eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'Eye.xml')
nose_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'nose.xml')
mouth_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'mouth.xml')
right_eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'right_eye.xml')
left_eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'left_eye.xml')

align = openface.AlignDlib(dlib_face_predictor_path)
# read torch net 128
net = openface.TorchNeuralNet(net_path_128, imgDim=96, cuda=False)
# svm classifier
with open(svm_path, 'r') as f:
    (le, clf) = pickle.load(f)  # le - label and clf - classifer

dlib_points = [[33,15], [67,15], [48,39]]

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
        return {'eye': eye[0], 'nose': nose[0], 'mouth': []}
    else:
        return None


def opencv_search_eyes(gray_img, scale_factor=1.1, min_neighbors=3):
    rigth_eye = right_eye_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)
    left_eye = left_eye_cascade.detectMultiScale(gray_img, scale_factor, min_neighbors)
    return {'r_eye': rigth_eye, 'l_eye': left_eye}


def opencv_find_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    print 'cascade face {}'.format(time.time() - start_time)
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

    return [[lx + w/2, ly + h], [rx + w/2, ry]]


def draw_rect(rect_dist, img):
    for name, rect in rect_dist.iteritems():
        if len(rect) > 0:
            x, y, w, h = rect
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, '{}'.format(name), (x+w/2, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_rect_in_list(rects, img):
    for rect in rects:
        x, y, w, h = rect
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)


def draw_points(points, img):
    for point in points:
        cv2.circle(img, (point[0], point[1]), 1, (255, 255, 0))


def predict_align_face(align_face):
    rep = net.forward(align_face)
    predictions = clf.predict_proba(rep).ravel()
    # print predictions
    maxI = np.argmax(predictions)
    # max2 = np.argsort(predictions)[-3:][::-1][1]
    return le.inverse_transform(maxI)


def predict_face(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    cv_senses = opencv_search_senses(gray_face)
    nose_point = get_nose_point(cv_senses['nose'])
    cv_eye_points = get_eye_points(cv_senses['eye'])
    cv_eye_points.append(nose_point)
    # affine transformation
    h, w, ch = face.shape
    # points
    pts1 = np.float32(cv_eye_points)
    pts2 = np.float32(dlib_points)
    M = cv2.getAffineTransform(pts1, pts2)
    cv_dlib = cv2.warpAffine(face, M, (w, h))

    name = predict_align_face(cv_dlib)
    return name, cv_eye_points

def opencv_find_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    person_info = []
    for face in faces:
        # get face image
        x, y, w, h = face
        roi = img[y:y + h, x:x + w]
        roi96 = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_CUBIC)
        # get name
        name, points = predict_face(roi96)
        person_info.append({'name': name, 'points': points, 'face_rect': face})
    return person_info

def dredict_and_draw(img):
    info = opencv_find_faces(img)
    draw_rect_in_list([info[0]['face_rect']], img)
    cv2.putText(img, info[0]['name'],
                (info[0]['face_rect'][0], info[0]['face_rect'][1] + info[0]['face_rect'][3]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)