import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()

        self.input_size = 6
        self.hidden_size = 256
        self.num_layers = 3
        self.output_size = 1
        self.threshold = 0.25

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

        self.apply(self.xavier_init) # use xavier initialization

    def xavier_init(self, m):
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input):
        if input.ndimension() == 1:
            input = input.unsqueeze(0)

        input = input.unsqueeze(0)

        outputs, _ = self.gru(input)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        outputs = self.sigmoid(outputs)
        outputs = outputs.view(-1)

        return outputs

    def evaluate(self, model, data_loader, loss_function, batch_size):
        total_loss = 0.0
        all_outputs = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for data in data_loader.get_test_data(batch_size):
                inputs, labels = data['input'], data['label']
                outputs = model(inputs)
                all_outputs.append(outputs)
                all_labels.append(labels)
                loss = loss_function(outputs, labels)
                total_loss += loss.item() * len(outputs)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        def f1_neg(threshold):
            predicted = (all_outputs > threshold).to(torch.float32)
            return -f1_score(all_labels.cpu().numpy(), predicted.cpu().numpy())
        self.threshold = minimize_scalar(f1_neg, bounds=(0, 1), method='bounded').x

        predicted_labels = (all_outputs > self.threshold).to(torch.float32)
        f1 = f1_score(all_labels.cpu().numpy(), predicted_labels.cpu().numpy())
        fn = ((all_labels == 0) & (predicted_labels == 1)).int().sum().item()
        fp = ((all_labels == 1) & (predicted_labels == 0)).int().sum().item()

        avg_loss = total_loss / len(data_loader.test_subset)
        return avg_loss, fn, fp, f1

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
