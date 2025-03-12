import os
import shutil
import pandas as pd

train_images_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\train_images"
train_labeled_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\data_fin\train\labeled"
train_unlabeled_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\data_fin\train\unlabeled"
val_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\data_fin\validation"
csv_file_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\train.csv"

df = pd.read_csv(csv_file_path)

class_counts1 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
class_counts2 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
class_counts3 = 0
max_counts1 = {0: 50, 1: 100, 2: 50, 3: 50, 4: 50}
max_counts2 = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}
max_counts3 = 100

op = 1
k1 = 0
k2 = 0
for idx, row in df.iterrows():
    image_name = row['image_id']
    label = row['label']

    if op == 1:
        if class_counts1[label] == max_counts1[label]:
            k1 += 1
            if k1 == 5:
                op = 2

        source_path = os.path.join(train_images_path, image_name)
        destination_path = os.path.join(train_labeled_path, f'{label:02}', image_name)

        shutil.copy(source_path, destination_path)

        class_counts1[label] += 1
    elif op == 2:
        if class_counts2[label] == max_counts2[label]:
            k2 += 1
            if k2 == 5:
                op = 3

        source_path = os.path.join(train_images_path, image_name)
        destination_path = os.path.join(val_path, f'{label:02}', image_name)

        shutil.copy(source_path, destination_path)

        class_counts2[label] += 1
    elif op == 3:
        if class_counts3 == max_counts3:
            op = 4

        source_path = os.path.join(train_images_path, image_name)
        destination_path = os.path.join(train_unlabeled_path, image_name)

        shutil.copy(source_path, destination_path)

        class_counts3 += 1
    else:
        print("图片复制完成")
        break
