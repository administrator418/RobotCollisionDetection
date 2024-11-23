import torch
import torch.nn as nn
import torch.nn.init as init

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()

        self.input_size = 6
        self.embed_size = 64
        self.hidden_size = 256
        self.num_heads = 8
        self.num_layers = 1
        self.output_size = 1
        self.threshold = 0.25

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

        # 模块定义
        self.input_embedding = nn.Linear(self.input_size, self.embed_size)  # 输入嵌入层
        self.position_embedding = nn.Embedding(100, self.embed_size)  # 位置编码（假设最大序列长度为100）

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=self.num_heads,
                                                   dim_feedforward=self.hidden_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Dropout 和全连接层
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.embed_size, self.output_size)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if input.ndimension() == 1:
            input = input.unsqueeze(0)
        input = input.unsqueeze(0)

        batch_size, seq_len, _ = input.size()

        # 生成位置索引并嵌入
        positions = torch.arange(seq_len, device=input.device).unsqueeze(0).repeat(batch_size, 1)
        embedded_input = self.input_embedding(input) + self.position_embedding(positions)

        # 转换形状以适配 Transformer: (seq_len, batch_size, embed_size)
        embedded_input = embedded_input.permute(1, 0, 2)

        # Transformer 编码器
        transformer_output = self.transformer(embedded_input)  # (seq_len, batch_size, embed_size)

        # 取最后一个时间步的输出
        final_output = transformer_output[-1]  # (batch_size, embed_size)

        # Dropout 和全连接层
        output = self.dropout(final_output)
        output = self.fc(output)
        output = self.sigmoid(output)

        return output.view(-1)

        return output

    def evaluate(self, model, data_loader, loss_function, batch_size):
        model.eval()

        total_loss = 0.0
        fns, fps, tps = 0, 0, 0

        with torch.no_grad():
            for data in data_loader.get_test_data(batch_size):
                inputs, labels = data['input'], data['label']

                output = model(inputs)
                loss = loss_function(output, labels)
                total_loss += loss.item() * len(output)

                output = (output > self.threshold).to(torch.float32)
                fns += ((output == 0) & (labels == 1)).sum().item()
                fps += ((output == 1) & (labels == 0)).sum().item()
                tps += ((output == 0) & (labels == 0)).sum().item()

        avg_loss = total_loss / len(data_loader.test_subset)
        return avg_loss, fns, fps, self.calculate_f1(tps, fps, tps)

    def calculate_f1(self, fn, fp, tp):
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
