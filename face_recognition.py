import cv2
import os
import pickle
import time
import copy

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface

dlib_face_predictor_path = "/home/timpfey/Work/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
net_path_128 = "/home/timpfey/Work/openface/models/openface/nn4.small2.v1.t7"
svm_path = "/home/timpfey/Work/openface/generated-embeddings/classifier.pkl"
im_path = "/home/timpfey/Work/openface/test-images/Screenshot_20161213_014837.png"
opencv_cascade_path = '/home/timpfey/Work/opencv/data/haarcascades/'

def draw_rectangle(rects, img, color):
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def dlib_align(img):
    # get reps
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get points
    start_time = time.time()
    # detect face
    bb = align.getAllFaceBoundingBoxes(rgb_img)
    print ("detect face {}".format(time.time() - start_time))

    # face detect opencv
    face_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'Nariz.xml')
    mouth_cascade = cv2.CascadeClassifier(opencv_cascade_path + 'Mouth.xml')

    # get aligned faces
    alignedFaces = []
    i = 0
    for box in bb:
        start_time = time.time()
        alignedFaces.append(
            align.align(
                96,
                rgb_img,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
        print ("align face {}".format(time.time() - start_time))
        cv2.imshow("{}".format(i), alignedFaces[i])
        cv2.imwrite("/home/timpfey/Work/openface/test-images/kat{}.png".format(i), alignedFaces[i])
        print (len(alignedFaces[i]))
        i += 1

    # open cv detect
    img_al = copy.copy(alignedFaces[0])
    gray = cv2.cvtColor(img_al, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    eyes = eye_cascade.detectMultiScale(gray)
    noses = nose_cascade.detectMultiScale(gray)
    mouth = mouth_cascade.detectMultiScale(gray)
    print ('detect face opencv {}'.format(time.time() - start_time))
    draw_rectangle(eyes, img_al, (255, 0, 0))
    draw_rectangle(noses, img_al, (0, 255, 0))
    draw_rectangle(mouth, img_al, (255, 255, 0))
    cv2.imshow('opencv', img_al)

    reps = []
    for alignedFace in alignedFaces:
        start_time = time.time()
        rec = net.forward(alignedFace)
        print("generate rec {}".format(time.time() - start_time))
        reps.append(rec)

    with open(svm_path, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
        start_time = time.time()
        predictions = clf.predict_proba(rep).ravel()
        print ("Person predict {}".format(time.time() - start_time))
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])

if __name__ == "__main__":
    img = cv2.imread(im_path)
    # read dlib model
    align = openface.AlignDlib(dlib_face_predictor_path)
    # read torch net 128
    net = openface.TorchNeuralNet(net_path_128, imgDim=96, cuda=False)

    cv2.imshow("origin", img)
    dlib_align(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



