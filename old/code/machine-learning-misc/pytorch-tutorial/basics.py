import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

#-------------------------------------------------------------------
print('Example 1: Basic autograd')

# Create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computation graph
y = w * x + b # y = 2 * x + 3

# Compute gradients
y.backward()

# Print out the gradients
print(x.grad) # dy/dx = 2
print(w.grad) # dy/dw = 1
print(b.grad) # dy/db = 1

#-------------------------------------------------------------------
print('Example 2: Backprop autograd')

# Create tensors of shape (10, 3) and (10, 2)
x = torch.rand(10, 3)
y = torch.rand(10, 2)

# Build a fully connected layer
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# Build loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass
pred = linear(x)

# Compute loss
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass
loss.backward()

# Print out the gradients
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

#Print out the loss after 1-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

#-------------------------------------------------------------------
print('Example 3: Numpy to tensor')

# Create a numpy array
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()

#-------------------------------------------------------------------
print('Example 4: Input pipeline')

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass

#-------------------------------------------------------------------
print('Example 5: Custom dataset')

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)

#-------------------------------------------------------------------
print('Example 5: Pretrained model')

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)

#-------------------------------------------------------------------
print('Example 6: Load and save')
# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
