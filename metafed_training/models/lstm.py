import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=52, hidden_size=128, num_classes=12, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.bn(h_n[-1])
        return self.fc(out)
