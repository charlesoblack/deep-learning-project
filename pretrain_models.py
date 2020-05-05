#! /usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data_helper import UnlabeledDataset, LabeledDataset


class PreTaskEncoder(nn.Module):
    def __init__(self, n_features):
        super(PreTaskEncoder, self).__init__()
        # number of different kernels to use
        self.n_features = n_features
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=n_features,
                               kernel_size=5,
                               )
        self.conv2 = nn.Conv2d(n_features,
                               int(n_features/2),
                               kernel_size=5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # return an array shape
        x = x.view(-1, int(26718/2))
        return x


# pretty much stolen from:
# https://github.com/Atcold/pytorch-Deep-Learning/blob/master/06-convnet.ipynb
class TestNet(torch.nn.Module):
    def __init__(self, n_features):
        super(TestNet, self).__init__()
        self.n_features = n_features
        self.encoder = PreTaskEncoder(n_features)
        self.fc1 = nn.Linear(int(26718/2), 50)
        self.fc2 = nn.Linear(50, 6)
    
    def forward(self, x):
        # encode image
        x = self.encoder(x)

        # decode and figure out which camera the image is from
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    # test the architecture

    np.random.seed(0)
    torch.manual_seed(0)

    image_folder = 'data'
    annotation_csv = 'data/annotation.csv'

    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 134)

    # stolen straight from the examples
    transform = torchvision.transforms.ToTensor()

    unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                          scene_index=unlabeled_scene_index,
                                          first_dim='image',
                                          transform=transform)
    # for some reason, with num_workers > 0 this always seems to fail with an 
    # "Empty" error from pytorch. found nothing online about it. :/
    trainloader = torch.utils.data.DataLoader(unlabeled_trainset,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # these train/test functions were also adapted from:
    # https://github.com/Atcold/pytorch-Deep-Learning/blob/master/06-convnet.ipynb
    def train(epoch, model, train_loader):
        model.to(device)
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        train_size = len(train_loader.dataset)
        
        for n in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                # send to device
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {n + 1:02d} '
                          f'[{batch_idx * len(data):05d}/{train_size}'
                          f' ({100 * batch_idx / len(train_loader):.0f}%)]'
                          f'\tLoss: {loss.item():.6f}')
                torch.cuda.empty_cache()


    def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_size = len(test_loader.dataset)
        mistakes = []
        torch.cuda.empty_cache()
        
        # no need for enumerate() because we don't use batch_idx
        for data, target in test_loader:
            # send to device
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]

            # change target shape to equal preds
            
            num_correct_predictions = (pred.eq(target.data.view_as(pred))
                                           .cpu()
                                           .sum()
                                           .item()
                                           )
            correct += num_correct_predictions

            torch.cuda.empty_cache()
            
            if num_correct_predictions < 64:
                wrong_ones = (target.data.view_as(pred) != pred).view(-1, )
                mistakes.append((data[wrong_ones],
                                 target[wrong_ones],
                                 pred[wrong_ones],
                                 ))

        test_loss /= test_size
        accuracy = 100. * correct / test_size

        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{test_size} ({accuracy:.0f}%)')

        return mistakes


    cnn = TestNet(6)
    train(2, cnn, trainloader)

    torch.save(cnn.encoder.state_dict(), 'pretrain_model_2_epochs.pt')
    
    assert cnn.__repr__() == 'TestNet(\n  (encoder): PreTaskEncoder(\n    (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))\n    (conv2): Conv2d(6, 3, kernel_size=(5, 5), stride=(1, 1))\n  )\n  (fc1): Linear(in_features=13359, out_features=50, bias=True)\n  (fc2): Linear(in_features=50, out_features=6, bias=True)\n)'

    transform = torchvision.transforms.ToTensor()

    labeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                        scene_index=labeled_scene_index,
                                        first_dim='image',
                                        transform=transform)
    testloader = torch.utils.data.DataLoader(labeled_trainset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=0)

    mistakes = test(cnn, testloader)

    assert len(mistakes) < 100

    
