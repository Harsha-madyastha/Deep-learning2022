import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_one_epoch(model,epoch_index, tb_writer, optimizer,loss_fn):
    running_loss = 0.
    last_loss = 0.

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    batch_sizes = [128, 256]
    for i in range(len(batch_sizes)):
        training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_sizes[i], shuffle=True, num_workers=2)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(model, EPOCHS, optimizer,name, lr):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # tracking=f'{optimizer}_{EPOCHS}_{lr}'
    writer = SummaryWriter(f"runs/fashion_trainer_{name}")
    loss_fn = torch.nn.CrossEntropyLoss()

    best_vloss = 1_000_000.
    epoch_number = 0
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model,epoch_number, writer,optimizer,loss_fn)

        # We don't need gradients on to do reporting
        model.train(False)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=2)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

def main():
    model = GarmentClassifier()
    lr=0.001
    optimize = "ADAM"
    if optimize == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
    train(model, 10, optimizer, optimize, lr)

if __name__=="__main__":
    main()
