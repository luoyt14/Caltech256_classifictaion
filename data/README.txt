Caltech 256数据集可在https://cloud.tsinghua.edu.cn/f/18de327151a14f4fa8c2/或http://www.vision.caltech.edu/Image_Datasets/Caltech256/下载。

train.txt, test.txt以及test_neg.txt分别表示训练集、测试集、开集测试集所对应的文件名及类别。
其中的每一行格式如下：
FileName/ImgName	Type

其中：
FileName为字符串，表示文件夹名称
ImgName为字符串，表示该图片的名称
Type为整数，表示图像类别，可能的取值包括0~20，0为背景clutter