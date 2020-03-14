import os
import numpy as np


IMAGE_ROOT = '../256_ObjectCategories/'
DATA_ROOT = '../data_additional/'
TEST_NUM = 25
print(len(os.listdir(IMAGE_ROOT)))


def create_filename(file_dir, num):
    label = file_dir.split('.')[0]
    num = str(num)
    image_name = file_dir + '/' + label + '_' + num.zfill(4) + '.jpg'
    if os.path.isfile(os.path.join(IMAGE_ROOT, image_name)):
        return True, file_dir + '/' + label + '_' + num.zfill(4) + '.jpg ' + str(int(label)-1)
    else:
        return False, ""


def create_train_test():
    test_image_list = []
    train_image_list = []
    for file_dir in os.listdir(IMAGE_ROOT):
        print('processing ' + file_dir)
        if not os.path.isdir(IMAGE_ROOT+file_dir):
            continue
        total_num = len(os.listdir(IMAGE_ROOT+file_dir))
        test_num = np.random.randint(0, total_num, size=(TEST_NUM,))
        for i, image in enumerate(os.listdir(IMAGE_ROOT+file_dir)):
            if i+1 in test_num:
                isfile, test_file_dis = create_filename(file_dir, i+1)
                if isfile:
                    test_image_list.append(test_file_dis)
            else:
                isfile, train_file_dis = create_filename(file_dir, i+1)
                if isfile:
                    train_image_list.append(train_file_dis)
    
    return train_image_list, test_image_list


def write_image_list(image_list, filename):
    with open(filename, 'w') as f:
        for item in image_list:
            f.write(item+'\n')


def write_train_test(train_image_list, test_image_list):
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    train_file_name = DATA_ROOT + 'train.txt'
    test_file_name = DATA_ROOT + 'test.txt'
    write_image_list(train_image_list, train_file_name)
    write_image_list(test_image_list, test_file_name)


if __name__ == '__main__':
    train_image_list, test_image_list = create_train_test()
    write_train_test(train_image_list, test_image_list)
