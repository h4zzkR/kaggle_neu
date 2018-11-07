from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
import torch
import torchvision
import pandas as pd
from torch.autograd import Variable
import torchvision.transforms as transforms

transform = transforms.Compose(

    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

)

trainset = torchvision.datasets.FashionMNIST(root='./data',
                                        train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                           shuffle=True, num_workers=0, pin_memory=False)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=0, pin_memory=False)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
COLOR_CHANNELS = 1
EPOCHS = 40
LEARNING_RATE = 0.001
BATCH_SIZE = 32
BATCH_IMAGE_COUNT = 10000
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']
N_CLASSES = len(CLASSES)
PLOT = False


def imshow(img):
    img = img / 2 + 0.5     # деномрировка
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_hidden_nodes, n_hidden_layers, keep_rate=0):
        super().__init__()
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        if not keep_rate:
            keep_rate = 0.4
        self.keep_rate = keep_rate
        # Set up perceptron layers and add dropout
        self.fc1 = torch.nn.Linear(IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS,
                                   n_hidden_nodes)
        self.fc1_drop = torch.nn.Dropout(1 - keep_rate)

        self.out = torch.nn.Linear(n_hidden_nodes, N_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS)
        sigmoid = torch.nn.modules.activation.Tanh()
        x = sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        return torch.nn.functional.log_softmax(self.out(x))


def train(epoch, model, train_loader, optimizer, log_interval=100, cuda=None):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        accuracy = 100. * correct / len(train_loader.dataset)

        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0], accuracy))


def validate(loss_vector, accuracy_vector, model, validation_loader, cuda=None):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += torch.nn.functional.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


def tests(model, loss_vector, val_load):
    # TESTS
    # Plot train loss and validation accuracy vs epochs for each learning rate
    epochs = [i for i in range(1, 41)]
    plt.plot(epochs, loss_vector)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    dataiter = iter(val_load)
    images, labels = dataiter.next()

    # выведем наши предсказания
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % CLASSES[labels[j]] for j in range(4)))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % CLASSES[predicted[j]]
                                  for j in range(4)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            CLASSES[i], 100 * class_correct[i] / class_total[i]))

    return pd.DataFrame(data=predicted.numpy(), columns=['Category'])


def main():
    hidden_nodes = 15
    layers = 1
    model = Net(hidden_nodes, layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_vector = []
    acc_vector = []
    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, train_loader, optimizer, cuda=False)
        validate(loss_vector, acc_vector, model, validation_loader, cuda=False)
        if epoch == 40:
            break

    answer_df = tests(model, loss_vector)
    answer_df.to_csv('./baseline.csv', index=False)


    TEST_PATH = './fashionmnist/fashion-mnist_test.csv'
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.values[:, 1:]  # удаляем столбец 'label'
    imagenet_data = torchvision.datasets.ImageFolder(TEST_PATH)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=False)
    torchvision.transforms.

    test_data = torch.FloatTensor(X_test)
    y_test_pred = net(test_data)
    _, predicted = torch.max(y_test_pred, 1)
    answer_df = pd.DataFrame(data=predicted.numpy(), columns=['Category'])
    answer_df.to_csv('./baseline.csv', index=False)

main()











