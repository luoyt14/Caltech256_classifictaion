import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io as scio
import logging
logging.basicConfig(level=logging.INFO)
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import time
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
np.random.seed(2019)


TRAIN_DATA_PATH = '../data/train.txt'
TEST_DATA_PATH = '../data/test.txt'
IMAGE_DATA_PATH = '../256_ObjectCategories/'


def process_image(image_name):
    img = Image.open(image_name)
    img = img.resize((224, 224))
    x_data = np.array(img) / 255.0

    if len(x_data.shape) != 3:
        temp = np.zeros((x_data.shape[0], x_data.shape[1], 3))
        temp[:, :, 0] = x_data
        temp[:, :, 1] = x_data
        temp[:, :, 2] = x_data
        x_data = temp

    image_feature = x_data.flatten()
    return image_feature


def compute_flase_alarm(ytrue, ypred):
    return np.sum(ytrue!=ypred) / (19 * ytrue.shape[0])


def read_file(filename):
    x_data = []
    labels = []
    with open(filename) as f:
        for line in f:
            image_name = line.split()[0]
            image_label = int(line.split()[1])
            image_path = IMAGE_DATA_PATH + image_name
            if os.path.isfile(image_path):
                image_feature = process_image(IMAGE_DATA_PATH + image_name)
                x_data.append(image_feature)
                labels.append(image_label)

    return np.array(x_data), np.array(labels)


def save_processed_data(X_train, X_test, y_train, y_test, ispca=""):
    XtrainName = "../data/Xtrain"+ispca+".mat"
    XtestName = "../data/Xtest"+ispca+".mat"
    YtrainName = "../data/Ytrain"+ispca+".mat"
    YtestName = "../data/Ytest"+ispca+".mat"
    scio.savemat(XtrainName, {'data': X_train})
    scio.savemat(XtestName, {'data': X_test})
    scio.savemat(YtrainName, {'data': y_train})
    scio.savemat(YtestName, {'data': y_test})


def load_processed_data(XtrainName, XtestName, YtrainName, YtestName):
    X_train = scio.loadmat(XtrainName)['data']
    X_test = scio.loadmat(XtestName)['data']
    y_train = scio.loadmat(YtrainName)['data']
    y_test = scio.loadmat(YtestName)['data']
    return X_train, X_test, y_train, y_test


def pca_decomposition(X_train, X_test, n_components=150):
    start = time.time()
    model = PCA(n_components=n_components, svd_solver='auto')
    model.fit(X_train)
    Xtrain = model.transform(X_train)
    Xtest = model.transform(X_test)
    logging.info(f'reduced demendion is {model.components_.shape[0]}')
    logging.info(f'reduced train mat shape is {Xtrain.shape}')
    logging.info(f'pca cost {time.time() - start} seconds')
    return Xtrain, Xtest


def knn_classify(X_train, X_test, y_train, y_test, k=1):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print(accuracy_score(y_test, y_predict))
    sn.heatmap(confusion_matrix(y_test, y_predict), annot=True)
    plt.show()
    return accuracy_score(y_test, y_predict), compute_flase_alarm(y_test, y_predict)

    
if __name__ == '__main__':
    # base analysis
    X_train, y_train = read_file(TRAIN_DATA_PATH)
    X_test, y_test = read_file(TEST_DATA_PATH)
    n_components_list = [21] #range(1, 1000, 5) # [51] [31]
    k=5
    accuracy_list = []
    flase_alarm_list = []
    # save_processed_data(X_train, X_test, y_train, y_test)
    # XtrainName = "../data/Xtrain.mat"
    # XtestName = "../data/Xtest.mat"
    # YtrainName = "../data/Ytrain.mat"
    # YtestName = "../data/Ytest.mat"
    # X_train, X_test, y_train, y_test = load_processed_data(XtrainName, XtestName, YtrainName, YtestName)
    for n_components in n_components_list:
        Xtrain, Xtest = pca_decomposition(X_train, X_test, n_components=n_components)
        # save_processed_data(Xtrain, Xtest, y_train, y_test, ispca='_pca')
        # XtrainName = "../data/Xtrain_pca.mat"
        # XtestName = "../data/Xtest_pca.mat"
        # YtrainName = "../data/Ytrain_pca.mat"
        # YtestName = "../data/Ytest_pca.mat"
        # X_train, X_test, y_train, y_test = load_processed_data(XtrainName, XtestName, YtrainName, YtestName)
        acc, flase_alarm = knn_classify(Xtrain, Xtest, y_train, y_test, k=k)
        print(f'n_components={n_components}, k={k}, accuracy={acc}, False Alarm={flase_alarm}')
        accuracy_list.append(acc)
        flase_alarm_list.append(flase_alarm)
    
    # plt.figure()
    # plt.plot(n_components_list, accuracy_list, label='accuracy vs. n_components', color='r')
    # plt.xlabel('n_components')
    # plt.ylabel("accuracy")
    # plt.title("accuracy vs. n_components")
    # plt.savefig('acc_n_5.png')

    # plt.figure()
    # plt.plot(n_components_list, flase_alarm_list, label='False Alarm vs. n_components', color='green')
    # plt.xlabel('n_components')
    # plt.ylabel("False Alarm")
    # plt.title("False Alarm vs. n_components")
    # plt.savefig('falseAlarm_n_5.png')
