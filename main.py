import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import _pickle as pickle
import matplotlib.image as image
from sys import argv
import cv2

import utils  # import the package in the project
import load_data  # import the package in the project


def train(path):
    img_tr, label_tr = load_data.load_mnist(path, 'train')  # load training data from dataset

    ss = StandardScaler()  # instantiate the method of normalization
    img_tr = ss.fit_transform(img_tr)  # normalize the training image and calculate the mean and variance
    img_ts, label_ts = load_data.load_mnist(path, 'test')  # load testing data from dataset
    img_ts_norm = ss.transform(img_ts)  # normalize the testing image

    model = svm.LinearSVC()  # construct model with different kernel function, you can use svm.SVC(kernel = 'rbf')/svm.SVC(kernel = 'poly') of radial basis kernel function and polynomial kernel function
    model.fit(img_tr, label_tr)  # training

    pre_label = model.predict(img_ts_norm)  # obtain the result of model

    with open('./model_lin.pkl', 'wb') as file:  # or model_rbf.pkl/model_poly.pkl
        pickle.dump(model, file)  # save the model

    print('Accuracy:', np.sum(pre_label == label_ts) / label_ts.size)  # calculate the accurary, the sum(pre_label == label_ts) represent the num of correct and label_ts.size represents the total num
    print(classification_report(label_ts, pre_label))  # print the report with precision, recall, F1-Score, macro average AVG and weighted average AVG
    return img_ts, pre_label, label_ts


def test(path):
    img_ori = image.imread(path)  # read real image
    img_ori = cv2.resize(28, 28)  # reshape real image into 28*28 as same as the dateset
    img_real = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)  # transfer the channel of real image from 3 into 1
    img = img_real.reshape(1, 784)  # reshape real image into 28*28=784 vector
    ss = StandardScaler()  # normalization
    img = ss.fit_transform(img)  # normalization
    with open('./model_rbf.pkl', 'rb') as m:  # use the rb format reading model via relative path
        model = pickle.load(m)  # load pretrained-model

    predict_label = model.predict(img)  # obtain result
    return img_ori, predict_label


if __name__ == '__main__':
    if argv[1] == 'train':  # you can select different mode via command
        dataset_path = argv[2]  # read the path of dataset in the argument
        img_test, pre_label, label = train(dataset_path)  # train and test 
        utils.show_result(img_test, pre_label, label)
    elif argv[1] == 'validation':  # select the mode of validation
        img_path = argv[2]  # read the path of real image in the argument
        image_real, pre_label = test(img_path)  # validate on real image
        utils.show_real(image_real, pre_label)
