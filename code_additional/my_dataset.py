import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io, transform
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root_dir, data_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []
        with open(data_path) as f:
            for line in f:
                image = line.split()[0]
                label = int(line.split()[1])
                self.image_list.append(image)
                self.label_list.append(label)


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list[idx])
        label = self.label_list[idx]
        image = Image.open(img_path)
        image = image.convert('RGB')
        # image = np.asarray(image)
        if self.transform:
            image = self.transform(image)
        
        return (image, label)

# from torchvision import transforms
# transform_test = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
# face_dataset = MyDataset('../256_ObjectCategories/', '../data_additional/test.txt', transform_test)

# for i in range(len(face_dataset)):
#     image, label = face_dataset[i]
#     # image = np.array(image)
#     print(image.shape, label)
#     # plt.imshow(image)
#     # plt.show()
#     # break
