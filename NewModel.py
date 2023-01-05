import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#prepare data
training_data = datasets.FashionMNIST(root="dataset", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="dataset", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# print(len(train_dataloader.dataset))
# print(len(test_dataloader.dataset))
# print(len(test_dataloader))
# images, labels = next(iter(train_dataloader))
# print(images.size(), labels.size())
# print(labels[0].item())
# plt.imshow(images[0].squeeze())
# plt.show()
# for images, labels in train_dataloader:
#     print(images.shape)

# build model
class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten1 = nn.Flatten()
        self.linearstack = nn.Sequential(nn.Linear(28*28, 512),
                               nn.ReLU(),
                               nn.Linear(512,128),
                               nn.ReLU(),
                               nn.Linear(128, 10))
    def forward(self, x):
        flatten =self.flatten1(x)
        output = self.linearstack(flatten)
        return output
        # self.l1 = nn.Flatten()
        # self.l2 = nn.Linear(28*28, 512)
        # self.l3 = nn.Linear(512,128)
        # self.l4 = nn.Linear(128,10)
        #
        # def Forward(self,x):
        #     x =nn.ReLU(self.l1(x))
        #     x = nn.ReLU(self.l2(x))
        #     x = nn.ReLU(self.l3(x))
        #     output = nn.Softmax(self.l4(x))
        #     return output


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Neural_Network().to(device)
# print(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

def train_model(train_dataloader, model, loss, optimizer):
    for i, (image, label) in enumerate(train_dataloader):
        # forward pass
        y_predict = model(image)
        loss_value = loss(y_predict,label)

        # backpropagation
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        if i%100 == 0:
            print(f'iteration no = {i}, loss_values = {loss_value.item()}')

def test_model(test_dataloader, model, loss):
    size = len(test_dataloader.dataset)
    mini_batch = len(test_dataloader)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x,y in test_dataloader:
            prediction = model(x)
            test_loss +=loss(prediction,y)
            correct += (prediction.argmax(1) == y).sum().item()
        test_loss = test_loss/mini_batch
        accuracy = correct/size
        print(f'Acurracy:{accuracy*100}, Loss={test_loss}')





# train and test model
epochs = 3
for i in range(epochs):
    print(f'Epoch number:{i}\n......')
    train_model(train_dataloader,model, loss, optimizer)
    test_model(test_dataloader,model,loss)
    print("Done")

# save model
torch.save(model.state_dict(), 'model.pth')

#load model
model.load_state_dict(torch.load('model.pth'))

#deploy model
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
model.eval()
x,y = test_data[7][0], test_data[7][1]
with torch.no_grad():
    pred = model(x)
    prediction, actual = classes[pred[0].argmax(0)], [classes[y]]
    print(f'prediction = {prediction}, actual = {actual}')





















#train Model
#Test Model
# save model
# load model
# deploy model through new predictions



