from torchvision import models
from torch import nn
import torch

import CUB_200_2011_dataloader

class train_config:
    dataset_class = 200
    device = torch.device("cuda:0")
    train_lr = 0.003
    model = "resnet50"

CLASS, train_dataloader = CUB_200_2011_dataloader.train_data_load()
_, test_dataloader = CUB_200_2011_dataloader.test_data_load()

if train_config.model == "resnet50":
    net = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = net.fc.in_features

    # for param in net.parameters():
    #     param.requires_grad = False #False：

    net.fc = nn.Sequential(nn.Linear(num_ftrs, train_config.dataset_class),
                                nn.LogSoftmax(dim = 1))

net.to(train_config.device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = train_config.train_lr)

def test(model, test_loader):
    print("Testing accuracy...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data, label = data
            data, label = data.to(train_config.device), label.to(train_config.device)
            output = model(data)
            _, predicted = torch.max(output.data, dim = 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        accuracy = (100 * correct / total)
        print('Accuracy: %d %% \n' % accuracy)
        return accuracy

def train(epoch):
    for epo in range(epoch):
        total_train_step = 0
        for input, targets in train_dataloader:
            input, targets = input.to(train_config.device), targets.to(train_config.device)
            output = net(input)

            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 50 == 0:
                print("epoch: {}, train times of batch size：{}, Loss: {}".format(epo, total_train_step, loss.item()))
        accuracy = test(net, test_dataloader)

        # save checkpoint
        checkpoint = {"model_state_dict": net.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epo,
                      "accuracy": accuracy}
        path_checkpoint = "./checkpoint/lr_{}_checkpoint_{}_epoch_{}%_accuracy.pkl".format(train_config.train_lr, epo, int(accuracy))
        torch.save(checkpoint, path_checkpoint)

train(100)
