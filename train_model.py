from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


class Config:
    batch_size = 50
    no_epochs = 50
    loss_function = nn.BCELoss()
    lr = 1e-3

def train_model(no_epochs):
    data_loaders = Data_Loaders(config.batch_size)
    model = Action_Conditioned_FF()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, config.loss_function)

    for epoch_i in range(no_epochs):
        model.train()
        train_bar = tqdm(data_loaders.train_loader)
        for data in train_bar: # sample['input'] and sample['label']
            inputs, labels = data['input'], data['label']
            output = model(inputs).squeeze()
            loss = config.loss_function(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch_i + 1,
                                                                    no_epochs,
                                                                    loss)

        losses.append(model.evaluate(model, data_loaders.test_loader, config.loss_function))
        if losses[-1] < min_loss:
            torch.save(model.state_dict(), 'saved/saved_model.pkl')
            min_loss = losses[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, no_epochs + 1), losses, marker='o', color='b', label='Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    config = Config()
    train_model(config.no_epochs)
