import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


# class NormModelSM(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.fc_img_1 = nn.Linear(IMG_LEN, 512)
#         self.fc_img_2 = nn.Linear(512, 256)
#
#         self.fc_txt_1 = nn.Linear(TXT_LEN, 256)
#         self.fc_txt_2 = nn.Linear(256, 256)
#
#         self.fc1 = nn.Linear(512, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.out = nn.Linear(128, N_CLASSES)
#
#         self.dropout = nn.modules.Dropout(p=0.25)
#
#     def forward(self, inp_img, inp_txt):
#         x_img = F.relu(self.fc_img_1(inp_img))
#         x_img = self.dropout(x_img)
#         x_img = F.relu(self.fc_img_2(x_img))
#         x_img = self.dropout(x_img)
#
#         x_txt = F.relu(self.fc_txt_1(inp_txt))
#         x_txt = self.dropout(x_txt)
#         x_txt = F.relu(self.fc_txt_2(x_txt))
#         x_txt = self.dropout(x_txt)
#
#         x = torch.cat((x_img, x_txt), 1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#
#         x = F.softmax(self.out(x), dim=1)
#         return x


class NormModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_img_1 = nn.Linear(IMG_LEN, 512)
        self.fc_img_2 = nn.Linear(512, 256)

        self.fc_txt_1 = nn.Linear(TXT_LEN, 256)
        self.fc_txt_2 = nn.Linear(256, 256)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=0.25)

    def forward(self, inp_img, inp_txt):
        x_img = F.relu(self.fc_img_1(inp_img))
        x_img = self.dropout(x_img)
        x_img = F.relu(self.fc_img_2(x_img))
        # x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt_1(inp_txt))
        x_txt = self.dropout(x_txt)
        x_txt = F.relu(self.fc_txt_2(x_txt))
        # x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class TrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IMG_LEN + TXT_LEN, 64)
        self.dropout = nn.modules.Dropout(p=0.25)
        self.fc2 = nn.Linear(64, N_CLASSES)

    def forward(self, inp_img, inp_txt):
        x = torch.cat((inp_img, inp_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

#
# model = NormModel()
# print(model)
#
# X_img = torch.randn(IMG_LEN).view(-1, IMG_LEN)
# X_txt = torch.randn(TXT_LEN).view(-1, TXT_LEN)
# output = model(X_img, X_txt)
# print(output)
# print(output.shape)