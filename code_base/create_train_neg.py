import os
import numpy as np
import re


IMAGE_ROOT = '../256_ObjectCategories/'
DATA_ROOT = '../data/'
CATEGORIES_PATH = '../data/categories.txt'
TEST_NEG_FILE = '../data/test_neg.txt'
TRAIN_NEG_FILES = 150
CHOOSE_CATEGORIES = '257.clutter'
print(len(os.listdir(IMAGE_ROOT)))


def create_filename(file_dir, num):
    label = file_dir.split('.')[0]
    num = str(num)
    image_name = file_dir + '/' + label + '_' + num.zfill(4) + '.jpg'
    if os.path.isfile(os.path.join(IMAGE_ROOT, image_name)):
        return True, file_dir + '/' + label + '_' + num.zfill(4) + '.jpg ' + str(0)
    else:
        return False, ""


def read_test_neg():
    test_neg_iamges = []
    with open(TEST_NEG_FILE) as f:
        for line in f:
            file_num = int(re.split('[./_\s]', line)[3])
            test_neg_iamges.append(file_num)

    return test_neg_iamges


def create_train_neg():
    train_image_neg_list = []
    test_neg_iamges = read_test_neg()
    total_num = len(os.listdir(IMAGE_ROOT+CHOOSE_CATEGORIES))
    train_neg_num = np.random.randint(0, total_num, size=(TRAIN_NEG_FILES,))
    for i, image in enumerate(os.listdir(IMAGE_ROOT+CHOOSE_CATEGORIES)):
        if i+1 in test_neg_iamges:
            continue
        if i+1 in train_neg_num:
            isfile, train_neg_file_dis = create_filename(CHOOSE_CATEGORIES, i+1)
            if isfile:
                train_image_neg_list.append(train_neg_file_dis)
    
    
    return train_image_neg_list


def write_image_list(image_list, filename):
    with open(filename, 'w') as f:
        for item in image_list:
            f.write(item+'\n')


def write_train_test(train_image_neg_list):
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    train_neg_file_name = DATA_ROOT + 'train_neg.txt'
    write_image_list(train_image_neg_list, train_neg_file_name)


if __name__ == '__main__':
    train_image_neg_list = create_train_neg()
    write_train_test(train_image_neg_list)
