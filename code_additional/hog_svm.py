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
from sklearn.metrics import accuracy_score, confusion_matrix
import time
from sklearn.svm import SVC, LinearSVC
import seaborn as sn
from skimage import feature
np.random.seed(2019)


TRAIN_DATA_PATH = '../data_additional/train.txt'
TEST_DATA_PATH = '../data_additional/test.txt'
IMAGE_DATA_PATH = '../256_ObjectCategories/'


def process_image(image_name):
    img = Image.open(image_name)
    img = img.resize((64, 64))
    img = img.convert('RGB')
    x_data = np.array(img) / 255.0

    # if len(x_data.shape) != 3:
    #     temp = np.zeros((x_data.shape[0], x_data.shape[1], 3))
    #     temp[:, :, 0] = x_data
    #     temp[:, :, 1] = x_data
    #     temp[:, :, 2] = x_data
    #     x_data = temp

    image_feature = feature.hog(x_data, 
                                orientations=13,
                                block_norm='L1', 
                                pixels_per_cell=[8, 8], 
                                cells_per_block=[4, 4], 
                                visualize=False, 
                                transform_sqrt=True,
                                feature_vector=True)
    # print(image_feature.shape)
    return image_feature


def compute_flase_alarm(ytrue, ypred):
    return np.sum(ytrue!=ypred) / (19 * ytrue.shape[0])


def read_file(filename):
    x_data = []
    labels = []
    with open(filename) as f:
        for line in f:
            image_name = line.split()[0]
            image_label = line.split()[1]
            image_path = IMAGE_DATA_PATH + image_name
            if os.path.isfile(image_path):
                image_feature = process_image(IMAGE_DATA_PATH + image_name)
                x_data.append(image_feature)
                labels.append(image_label)

    return np.array(x_data), np.array(labels)


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

def svm_classify(X_train, X_test, y_train, y_test):
    start2 = time.time()
    model = LinearSVC()
    # model = SVC(max_iter=1000, gamma='scale', probability=True, kernel='linear')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print(f'accuracy={accuracy_score(y_test, y_predict)}')
    print(f'false alarm={compute_flase_alarm(y_test, y_predict)}')
    sn.heatmap(confusion_matrix(y_test, y_predict), annot=True)
    plt.savefig("svmtest.png")
    plt.show()


if __name__ == '__main__':
    X_train, y_train = read_file(TRAIN_DATA_PATH)
    X_test, y_test = read_file(TEST_DATA_PATH)
    # Xtrain, Xtest = pca_decomposition(X_train, X_test, n_components=700)
    svm_classify(X_train, X_test, y_train, y_test)

