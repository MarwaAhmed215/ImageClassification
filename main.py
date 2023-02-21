import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
random.seed(12)
torch.manual_seed(18)
path = "dataset/dogs-vs-cats"
train_dir = "train"
test_dir = "test1"
path1 = os.path.join(path,train_dir)
files_train = os.listdir(path1)

path2 = os.path.join(path, test_dir)
files_test = os.listdir(path2)

# print(np.size(files_train))
# print(files_train[:5][1])
labels_train = []
for filename in files_train:
    category = filename.split('.')[0]
    if category =='dog':
        labels_train.append(1)
    else:
        labels_train.append(0)

# labels_test = []
# for filename in files_test:
#     category = filename.split('.')[0]
#     if category =='dog':
#         labels_test.append(1)
#     else:
#         labels_test.append(0)
imgsize = (28,28)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
# transform = transforms.Resize(imgsize)
x_train, x_test, y_train, y_test = train_test_split(files_train, labels_train,
                                                    test_size=0.2, random_state=42)
df_train = pd.DataFrame({'filename':x_train, 'label':y_train})
df_test = pd.DataFrame({'filename': x_test, 'label':y_test})
# print(len(df_train))
# print(len(df_test))
# iterate over items rows
# for i, row in df_train.iterrows():
#     if i % 1000 == 0:
#         print(i, row['filename'], row['label'])

# df['label'] = df['label'].dtype('int')
# print(df.head())
# print(df['label'])
# df.info()
class CustomImageDataset(Dataset):
    def __init__(self,labels, imgDirectory, transforms=None):
        self.img_labels = labels
        self.imgdir = imgDirectory
        self.transform = transforms

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.imgdir, self.img_labels.iloc[index,0])
        image = Image.open(img_path)
        # print(image)
        size = image.size
        label = self.img_labels.iloc[index,1]
        label = np.array(label)
        label_tensor=torch.from_numpy(label)
        if self.transform is not None:
            image = self.transform(image)
        return image,label_tensor

training_data = CustomImageDataset(df_train,path1, train_transforms)
test_data = CustomImageDataset(df_test, path1, train_transforms)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# iterate over items rows
images, label = next(iter(train_dataloader))
# print(f"**************training_size:{images.size()}, ******************labels_size:{label.size()}")
# print(images[0])
# plt.imshow(images[0][0])
# plt.show()
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,stride=2,padding=0),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.l2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,stride=2,padding=0),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.l3 = nn.Sequential(nn.Conv2d(32,64, kernel_size=3,stride=2,padding=0),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.fc1 = nn.Linear(576,10)
        self.doupout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()

    def forward(self,x):
        # x = x.unsqueeze(0)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        # print("feature map_____________", out.shape)
        out = out.view(out.size(0),-1)
        # print("flatten feature map_____________", out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

device = 'gpu' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)
# print(model)
lossValue = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

def train_model(model, trainingData, loss, optimizer):
    for i, (x,y) in enumerate(trainingData):
        # forward pass
        y_predict = model(x)
        lossOut= loss(y_predict,y)
        #Backpropagation
        optimizer.zero_grad()
        lossOut.backward()
        optimizer.step()
        if i%100 == 0:
            critirion = lossOut.item()
            print(f'iteration number:{i}, Loss value ={critirion}')

def test_model(model, test_data, loss):
    size = len(test_dataloader.dataset)
    mini_batch_size = len(test_dataloader)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x,y in test_data:
            pred = model(x)
            test_loss += loss(pred,y).item()
            correct = correct + (pred.argmax(1)==y).sum().item()
    test_loss = test_loss/mini_batch_size
    accuracy = correct/size
    print(f'Test_loss \n accuracy:{accuracy*100}, loss={test_loss}')

#------------------- train Model and save it----------------------
epoches = 10
for i in range (epoches):
    print(f"epoch {i+1}\n-----")
    train_model(model, train_dataloader, lossValue, optimizer)
    test_model(model, test_dataloader, lossValue)
torch.save(model.state_dict(), 'catDogs.pth')
print("saved model dictionary to catDogs.pth")

# ----------------validating trained model------------
#load model
model = Model()

#load state dictionary
model.load_state_dict(torch.load('BestModel/catDogs.pth'))
print("model is -----------------------",model)

#create preprocessing transformation
valid_data = files_test
# print(valid_data)
image = os.path.join(path2,valid_data[541])
img = Image.open(image)
plt.imshow(img)
plt.show()
# img = np.asarray(img)
img = train_transforms(img)


#unsqueeze batch dimention in case you are dealing with single image
input = img.unsqueeze(0)

# set model to eval
classes = ["cat", "dog"]
model.eval()


# use model to make predictions
with torch.no_grad():
     pred = model(input)
     prediction = classes[pred[0].argmax(0)]
     print(f"predicted:{prediction}")


