import torch
import torch.nn as nn
import torchvision
from typing import Optional
from contextlib import nullcontext

class ResNetConcatRNN(nn.Module):
    def __init__(
        self,
        resnet: nn.Module,
        channel_dim: int,
        tabular_dim: int,
        out_dim: int,
        dropout_rate: float = 0.0,
        hidden_dim: Optional[int] = None, # dimension of RNN hidden state
        rnn_layers: int = 1,  # number of RNN layers
        rnn_type: str = 'GRU',  # type of RNN (GRU, LSTM, RNN)
        pretrain_type: str = None,
        freeze_enc=False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.resnet = modify_resnet_channels(resnet, channel_dim, pretrain_type)
        self.freeze_enc = freeze_enc
        if freeze_enc:
            self.resnet.requires_grad = False
        self.rnn_input_size = resnet.fc.in_features + tabular_dim

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.rnn_input_size, 
                hidden_size=hidden_dim, 
                num_layers=rnn_layers,
                batch_first=True
            )
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.rnn_input_size, 
                hidden_size=hidden_dim, 
                num_layers=rnn_layers,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.rnn_input_size, 
                hidden_size=hidden_dim, 
                num_layers=rnn_layers,
                batch_first=True
            )
        else:
            raise ValueError('Unsupported rnn type.')
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def get_head_parameters(self):
        return list(self.rnn.parameters()) + list(self.output_layer.parameters())
    
    def forward(self, img_seq, tab_seq):
        batch_size, seq_len, C, H, W = img_seq.size()
        _, _, num_tab_features = tab_seq.size()

        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_dim).to(img_seq.device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_dim).to(img_seq.device)
            rnn_state = (h0, c0)
        else:
            rnn_state = h0

        rnn_input = torch.zeros(batch_size, seq_len, self.rnn_input_size).to(img_seq.device)

        with torch.no_grad() if self.freeze_enc else nullcontext():
            resnet_output = self.process_through_resnet(img_seq)
        combined_input = torch.cat((resnet_output, tab_seq), dim=-1)
        combined_input = self.dropout_1(combined_input)
        rnn_input = combined_input

        rnn_output, _ = self.rnn(rnn_input, rnn_state)
        output = self.dropout_2(rnn_output)
        output = self.output_layer(output)
    
        return output
    
    def process_through_resnet(self, img_seq):
        """
        Pass a sequence of images through the ResNet layers.
        """
        batch_size, seq_len, C, H, W = img_seq.size()
        img_seq = img_seq.view(batch_size * seq_len, C, H, W)

        x = self.resnet.conv1(img_seq)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        _, resnet_feature_size = x.shape
        x = x.view(batch_size, seq_len, resnet_feature_size)
        return x

def modify_resnet_channels(resnet, channel_dim, pretrain_type):
    """
    Modify the first convolutional layer of ResNet to accommodate different channel dimensions.
    """
    conv1_weight_original = resnet.conv1.weight.clone()
    resnet.conv1 = nn.Conv2d(channel_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1_weight_new = resnet.conv1.weight.data.clone()

    if pretrain_type == 'imagenet':
        conv1_weight_new[:, :3] = conv1_weight_original
    elif pretrain_type == 'sent2-moco-all':
        conv1_rgb_weight = conv1_weight_original[:, [3, 2, 1], :, :].clone()
        conv1_weight_new[:, :3, :, :] = conv1_rgb_weight
        conv1_nir_weight = conv1_weight_original[:, 7, :, :].clone()
        conv1_weight_new[:, 3, :, :] = conv1_nir_weight
    elif pretrain_type == 'sent2-moco-rgb':
        conv1_rgb_weight = conv1_weight_original[:, [2, 1, 0], :, :].clone()
        conv1_weight_new[:, :3, :, :] = conv1_rgb_weight

    resnet.conv1.weight.data = conv1_weight_new

    return resnet

# def create_fc_layers(resnet, tabular_dim, hidden_dim, lstm_hidden_dim):
#     """
#     Create fully connected layers to process combined ResNet and tabular features.
#     """
#     resnet_emb_dim = resnet.fc.in_features
#     fc_layers = []
#     if hidden_dim is not None:
#         fc_layers.append(nn.Linear(resnet_emb_dim + tabular_dim, hidden_dim))
#         fc_layers.append(nn.GELU())
#         fc_layers.append(nn.Linear(hidden_dim, lstm_hidden_dim))
#     else:
#         fc_layers.append(nn.Linear(resnet_emb_dim + tabular_dim, lstm_hidden_dim))
#     return nn.ModuleList(fc_layers)