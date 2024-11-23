from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import torch_optimizer as optim
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('RobotCollisionDetection/saved/train_model.log')])

class Config:
    batch_size = 64
    no_epochs = 1000
    loss_function = nn.BCELoss()
    lr = 1e-3
    weight_decay = 1e-3
    k = 5
    alpha = 0.5
    ls_patience = 7
    ls_factor = 0.1

def train_model(no_epochs):
    data_loader = Data_Loaders()
    model = Action_Conditioned_FF()

    base_optimizer = optim.AdamP(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer = optim.Lookahead(base_optimizer, k=config.k, alpha=config.alpha)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.ls_factor, patience=config.ls_patience
    )

    train_losses = []
    test_losses = []
    min_loss, _, _, _ = model.evaluate(model, data_loader, config.loss_function, config.batch_size)

    for epoch_i in range(no_epochs):
        model.train()
        total_loss = 0.0
        for data in data_loader.get_train_data(config.batch_size): # sample['input'] and sample['label']
            inputs, labels = data['input'], data['label']
            outputs = model(inputs)
            loss = config.loss_function(outputs, labels)
            total_loss += loss.item() * len(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss, fn, fp, f1 = model.evaluate(model, data_loader, config.loss_function, config.batch_size)
        scheduler.step(test_loss)

        if epoch_i % 10 == 0:
            avg_loss = total_loss / len(data_loader.train_subset)
            train_losses.append(avg_loss)
            test_losses.append(test_loss)

            if test_loss < min_loss:
                torch.save(model.state_dict(), 'RobotCollisionDetection/saved/saved_model.pkl')
                with open('RobotCollisionDetection/saved/bast_threshold.txt', 'w') as file:
                    file.write(str(model.threshold))
                min_loss = test_loss

            logging.info(
                f'Epoch: {epoch_i + 1}, '
                f'train loss: {avg_loss:.3f}, '
                f'test loss: {test_loss:.3f}')
            logging.info(
                f'Threshold: {model.threshold:.3f}, '
                f'F1 Score: {f1:.3f}, '
                f'False Negative: {fn}/{len(data_loader.test_subset)}, '
                f'False Positive: {fp}/{len(data_loader.test_subset)}'
            )

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, no_epochs + 1), train_losses, marker='o', color='b', label='train loss')
    plt.plot(range(1, no_epochs + 1), test_losses, marker='o', color='r', label='test loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    image_path = 'RobotCollisionDetection/saved/loss.png'
    plt.savefig(image_path)
    plt.show()
    logging.info(f'Loss image saved to {image_path}')

if __name__ == '__main__':
    config = Config()
    train_model(config.no_epochs)
