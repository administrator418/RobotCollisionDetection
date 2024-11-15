import torch
import torch.nn as nn
import torch.nn.init as init

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()

        self.input_size = 6
        self.hidden_size = 128
        self.num_layers = 1
        self.output_size = 1

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.xavier_init) # use xavier initialization

    def xavier_init(self, m):
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, input):
        if input.ndimension() == 1:
            input = input.unsqueeze(0)

        input = input.unsqueeze(0)

        output, _ = self.gru(input)
        output = self.fc(output)
        output = self.sigmoid(output)
        output = output.squeeze()

        return output

    def evaluate(self, model, test_loader, loss_function):
        model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data['input'], data['label']

                output = model(inputs).squeeze()

                loss = loss_function(output, labels)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
