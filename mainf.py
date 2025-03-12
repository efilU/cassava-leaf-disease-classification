import random
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image  # 读取图片数据
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from model_utils.model import initialize_model

# 获取随机种子
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(0)  # 在0参数下随机的数据不会改变
# 即使你保存了模型参数，如果训练过程中存在随机性（如数据分割、数据增强、模型初始化等），
# 不同运行可能会产生不同的模型参数。

HW = 224  # 长宽

# 定义transform变换
train_transform = transforms.Compose(  # 对训练图片进行预处理变换
    [
        transforms.ToPILImage(),  # 如图片由尺寸224,224,3变为3,224,224
        transforms.RandomResizedCrop(224),  # 将图片尺寸裁切为3*224*224
        transforms.RandomRotation(50),  # 图片随机旋转0~50°
        transforms.ToTensor()  # 将图片转化为张量
    ]
)

val_transform = transforms.Compose(  # 对训练图片进行预处理变换
    [
        transforms.ToPILImage(),  # 如图片由尺寸224,224,3变为3,224,224
        transforms.ToTensor()  # 将图片转化为张量
    ]
)


# Dataset数据处理
class leaf_Dataset(Dataset):  # 创建类继承Dataset
    def __init__(self, path, mode="train"):  # 构造类函数，默认为训练模式
        self.mode = mode
        if mode == "semi":  # 半监督模式，不需要读入y
            self.X = self.read_file(path)  # 调用read_file函数
        else:
            self.X, self.Y = self.read_file(path)
            self.Y = torch.LongTensor(self.Y)  # 标签转化为长整型

        if mode == "train":
            self.transform = train_transform
        else:  # 包含测试集和半监督集
            self.transform = val_transform

    def read_file(self, path):
        if self.mode == "semi":
            file_list = os.listdir(path)  # 列出指定目录下的所有文件和子目录的名称
            X = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
                # 创建一个形状为 (文件数量, 高度, 宽度, 通道数) 的全零数组，数据类型为 uint8，适合存储图像数据
            for index, img_name in enumerate(file_list):  # 列出文件夹下所有文件名字
                img_path = os.path.join(path, img_name)  # 路径拼接
                img = Image.open(img_path)
                img = img.resize((HW, HW))
                X[index, ...] = img
            print("读到了%d个数据" % len(X))
            return X
        else:
            class_samples = []
            for i in tqdm(range(5)):  # tqdm使过程形象化
                file_dir = path + "/%02d" % i
                file_list = os.listdir(file_dir)  # 列出指定目录下的所有文件和子目录的名称

                xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
                yi = np.zeros(len(file_list), dtype=np.uint8)

                for index, img_name in enumerate(file_list):
                    img_path = os.path.join(file_dir, img_name)
                    img = Image.open(img_path)
                    img = img.resize((HW, HW))
                    xi[index, ...] = img
                    yi[index] = i  # 最终读到数据数组xi，标签数据yi

                class_samples.append(xi)
                class_samples.append(yi)

            X = np.concatenate(class_samples[::2], axis=0)  # 在第0维进行数组拼接
            Y = np.concatenate(class_samples[1::2], axis=0)
            print("读到了%d个数据" % len(Y))
            return X, Y

    def __getitem__(self, item):  # 按下边读数据
        if self.mode == "semi":
            return self.transform(self.X[item]), self.X[item]  # 返回变换后和变换前的数据
        else:
            return self.transform(self.X[item]), self.Y[item]  # 返回变换后的x，和对应的yi

    def __len__(self):
        return len(self.X)  # 返回数据总长度


# 半监督数据处理（用于判断无标签数据是否可用数据）
class semiDataset(Dataset):
    def __init__(self, no_label_loader, model, device, thres=0.99):
        x, y = self.get_label(no_label_loader, model, device, thres)
        if x == []:  # 没有可用数据的情况
            self.flag = False
        else:
            self.flag = True
            self.X = np.array(x)
            self.Y = torch.LongTensor(y)
            self.transform = train_transform

    def get_label(self, no_babel_loader, model, device, thres):
        model = model.to(device)
        pred_prob = []
        labels = []
        x = []
        y = []
        soft = nn.Softmax()  # 将模型的输出转化为概率分布（e^x）
        with torch.no_grad():  # 对semi数据不更新梯度
            for bat_x, _ in no_babel_loader:
                bat_x = bat_x.to(device)
                pred = model(bat_x)
                pred_soft = soft(pred)  # 对预测值进行处理
                pred_max, pred_index = pred_soft.max(1)  # 在第二个维度上获取最大概率及其下标
                pred_prob.extend(pred_max.cpu().numpy().tolist())
                labels.extend(pred_index.cpu().numpy().tolist())

        for index, prob in enumerate(pred_prob):  # 获取每一个预测最大概率及其在列表中的下表
            if prob > thres:  # 判断置信度
                x.append(no_babel_loader.dataset[index][1])  # 获取dataset传入的原始数据
                y.append(labels[index])  # 获取预测的类别
        return x, y

    def __getitem__(self, item):
        return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.X)

# 获取可用的半监督数据
def get_semi_loader(no_laber_loader, model, device, thres):
    semiset = semiDataset(no_laber_loader, model, device, thres)
    if semiset.flag == False:
        return None
    else:
        semi_loader = DataLoader(semiset, batch_size=16, shuffle=False)  # 半监督数据不要打乱
        return semi_loader

# 自定义加权交叉熵损失函数
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights, device):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = torch.tensor(weights).to(device)

    def forward(self, inputs, targets):
        loss = nn.functional.cross_entropy(inputs, targets, weight=self.weights)
        return loss

# 训练函数
def train_val(model, train_loader, val_loader, no_label_loader,
              device, epochs, optimizer, loss_fn, thres, save_path):
    model = model.to(device)
    semi_loader = None
    plt_train_loss = []
    plt_val_loss = []
    plt_train_acc = []
    plt_val_acc = []
    max_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        semi_loss = 0.0
        semi_acc = 0.0

        start_time = time.time()

        model.train()  # 训练模式
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_bat_loss = loss_fn(pred, target)  # 由超参定义的loss
            train_bat_loss.backward()
            optimizer.step()  # 起到更新模型的作用——Adamw
            optimizer.zero_grad()  # 梯度清零，避免累计
            train_loss += train_bat_loss.detach().cpu().item()  # .detach()不再参与梯度计算，.cpu()放到cpu上计算，.item()提取张量中单个python数字
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
                            # axis=1表示维度为1，atgmax表示维度1中最大值的下标，与target中的真实标签比较，得到准确率
        plt_train_loss.append(train_loss / train_loader.__len__())  # 记录本轮loss值，除数为批数
        plt_train_acc.append(train_acc / train_loader.dataset.__len__())  # 记录本轮准确率，除数为数据总数

        if semi_loader != None:
            for batch_x, batch_y in semi_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                semi_bat_loss = loss_fn(pred, target)
                semi_bat_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                semi_loss += semi_bat_loss.cpu().item()
                semi_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
            print("半监督数据集的训练准确率为", semi_acc / semi_loader.dataset.__len__())

        model.eval()  # 验证模式
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss_fn(pred, target)
                val_loss += val_bat_loss.cpu().item()
                val_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())
        plt_val_loss.append(val_loss / val_loader.dataset.__len__())
        plt_val_acc.append(val_acc / val_loader.dataset.__len__())

        if epoch % 3 == 0 and plt_val_acc[-1] > 0.65:  # 每隔三轮进行一次半监督检查
            semi_loader = get_semi_loader(no_label_loader, model, device, thres)

        if val_acc > max_acc:
            torch.save(model, save_path)
            max_acc = val_acc
            best_epoch = epoch

        print("[%03d/%03d] %2.2f sec(s) TrainLoss: %.6f | ValLoss: %.6f  Trainacc: %.6f | Valacc: %.6f" %
              (epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1],
              plt_train_acc[-1], plt_val_acc[-1]))

        # 早停法
        if epoch - best_epoch >= 5:
            print(f"Validation accuracy has not improved for 5 epochs. Early stopping.")
            break

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend(["train", "val"])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("acc")
    plt.legend(["train", "val"])
    plt.show()


train_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\data_fin\train\labeled"
val_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\data_fin\validation"
no_label_path = r"C:\Users\HP\Desktop\深度学习项目\cassava-leaf-disease-classification\data_fin\train\unlabeled"

train_set = leaf_Dataset(train_path, "train")
val_set = leaf_Dataset(val_path, "val")
no_label_set = leaf_Dataset(no_label_path, "semi")

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
no_label_loader = DataLoader(no_label_set, batch_size=16, shuffle=True)

# 初始化模型
model, _ = initialize_model("resnet18", 5, use_pretrained=True)

# 超参部分
lr = 0.0001  # 减小学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "model_save/best_model.pth"
epochs = 30  # 增加训练轮数
thres = 0.99

# 计算类别权重
class_sample_counts = [len(train_set.Y[train_set.Y == t]) for t in range(5)]
total_samples = sum(class_sample_counts)
weights = [total_samples / (5 * count) for count in class_sample_counts]

print(f"Class sample counts: {class_sample_counts}")
print(f"Weights: {weights}")

# 设置自定义加权交叉熵损失函数
loss_fn = WeightedCrossEntropyLoss(weights, device)

# 开始训练
train_val(model, train_loader, val_loader, no_label_loader, device, epochs, optimizer, loss_fn, thres, save_path)