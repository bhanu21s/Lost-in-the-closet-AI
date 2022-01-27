#importing libraries
import torchvision
import torchvision.transforms as transforms 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


#FahionMNISt traning and testing data
train_set = torchvision.datasets.FashionMNIST(root = ".", train = True, download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train = False, download = True, transform = transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False)
torch.manual_seed(0)



#convolution neural network model
class CNN_MODEL(nn.Module):

    def __init__(self):
        super(CNN_MODEL, self).__init__()
        
        #activation function
        self.activ_function = torch.relu 

        # 1 input image channel, 32 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)

        # Initiliazing with Xavier Uniform 
        nn.init.xavier_uniform_(self.conv1.weight)

        # 32 input image channel, 64 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 5)

        # layers are fully connected
        self.fullyconnected1 = nn.Linear(64 * 4 * 4, 1024)  # 4*4 from image dimension


        self.fullyconnected1 = nn.Linear(1024, 256)

        #output layer
        self.output = nn.Linear(256, 10)

    
        # Defining the proportion or neurons to dropout
        self.dropout = nn.Dropout(p = 1)

    def forward(self, x):
        # Max pooling over (2, 2) window
        x = torch.max_pool2d(self.self.activ_function(self.conv1(x)), (2, 2))

        # Max pooling over a (2, 2) window
        x = torch.max_pool2d(self.self.activ_function(self.conv2(x)), (2, 2))

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        # Fully connected layer
        x = self.activ_function(self.fc1(x))
        x = self.activ_function(self.fc2(x)) 
        
        #dropout layer
        x = self.dropout(x)
        
        #output layer
        x = self.output(x)
        return x


CNN_MODEL = CNN_MODEL()
print(CNN_MODEL)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CNN_MODEL.parameters(), lr= 0.1)


num_epochs = 50

train_accuracies = []
train_losses = []

for epoch in range(num_epochs):  # iterating over the dataset multiple times

    
    correct = 0
    total = 0
    train_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = CNN_MODEL(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # printing statistics
        train_loss += loss.item()

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    train_accuracies.append(accuracy)
    train_losses.append(train_loss)

    print('Epoch [{}] loss: {} accuracy: {} %'.format(epoch + 1, train_loss, accuracy))
        
print('Finished Training')


num_epochs = 50

test_accuracies = []
test_losses = []

for epoch in range(num_epochs):  # iterating over the dataset multiple times

    correct = 0
    total = 0
    test_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = CNN_MODEL(images)
        loss = criterion(outputs, labels)
       
        # predicting the class with highest energy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics
        test_loss += loss.item()
    
    accuracy = 100 * correct / total

    test_accuracies.append(accuracy)
    test_losses.append(test_loss)

    print('Epoch {} loss: {} accuracy: {} %'.format(epoch + 1, test_loss, accuracy))
print('Finish Testing')


#plotting 
plt.plot(range(1,51),train_accuracies,label='Train')
plt.plot(range(1,51),test_accuracies,label='Test')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('train_test_accuracy_plot.png')
plt.show()

plt.plot(range(1,51),train_losses,label='Train')
plt.plot(range(1,51),test_accuracies,label='Test')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_test_losses_plot.png')
plt.show()