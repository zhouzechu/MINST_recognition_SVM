import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import _pickle as pickle
import utils
import matplotlib.image as image
from sys import argv
import cv2
import load_data


def train(path):
    img_tr, label_tr = load_data.load_mnist(path, 'train')

    ss = StandardScaler()
    img_tr = ss.fit_transform(img_tr)
    img_ts, label_ts = load_data.load_mnist(path, 'test')
    img_ts_norm = ss.transform(img_ts)

    model = svm.LinearSVC()
    model.fit(img_tr, label_tr)

    pre_label = model.predict(img_ts_norm)

    with open('./model_lin.pkl', 'wb') as file:
        pickle.dump(model, file)

    print('Accuracy:', np.sum(pre_label == label_ts) / label_ts.size)
    print(classification_report(label_ts, pre_label))
    return img_ts, pre_label, label_ts


def test(path):
    img_ori = image.imread(path)
    img_ori = img_ori.reshape(28, 28)
    img_real = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    img = img_real.reshape(1, 784)
    ss = StandardScaler()
    img = ss.fit_transform(img)
    with open('./model_rbf.pkl', 'rb') as m:
        model = pickle.load(m)

    predict_label = model.predict(img)
    return img_ori, predict_label


if __name__ == '__main__':
    if argv[1] == 'train':
        img_test, pre_label, label = train('C:/Users/PC/Desktop/SVM/')
        utils.show_result(img_test, pre_label, label)
    elif argv[1] == 'test':
        image_real, pre_label = test('C:/Users/PC/Desktop/SVM/4.png')
        utils.show_real(image_real, pre_label)
