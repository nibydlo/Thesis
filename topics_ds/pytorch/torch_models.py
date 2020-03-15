import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


def fit_topics_model(model, optimizer, train_loader, val_loader, scheduler=None, writer=None, epochs=1):
    for epoch in range(epochs):
        model.train()

        loss_sum = 0.0
        loss_count = 0

        for x_img_cur, x_txt_cur, y_cur in train_loader:
            model.zero_grad()
            output = model(x_img_cur.view(-1, IMG_LEN).float(), x_txt_cur.view(-1, TXT_LEN).float())
            loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
            loss.backward()

            loss_sum += loss
            loss_count += 1

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        print('epoch:', epoch, 'train_loss:', loss, 'average train loss', loss_sum / loss_count)
        if writer is not None:
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('avg_train_loss', loss_sum / loss_count, epoch)

        if val_loader is not None:
            model.eval()

            correct = 0
            total = 0
            loss_sum = 0.0
            loss_count = 0

            with torch.no_grad():
                for x_img_cur, x_txt_cur, y_cur in val_loader:
                    output = model(x_img_cur.view(-1, IMG_LEN).float(), x_txt_cur.view(-1, TXT_LEN).float())
                    loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
                    loss_sum += loss
                    loss_count += 1
                    for idx, i in enumerate(output):
                        if torch.argmax(i) == torch.argmax(y_cur, dim=1)[idx]:
                            correct += 1
                        total += 1

            print('val_acc:', correct / total, 'val_avg_loss:', loss_sum / loss_count)
            if writer is not None:
                writer.add_scalar('val_acc', correct / total, epoch)
                writer.add_scalar('val_avg_loss', loss_sum / loss_count, epoch)


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
        x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt_1(inp_txt))
        x_txt = self.dropout(x_txt)
        x_txt = F.relu(self.fc_txt_2(x_txt))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class NormModelGlorot(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_img_1 = nn.Linear(IMG_LEN, 512)
        nn.init.xavier_uniform_(self.fc_img_1.weight)

        self.fc_img_2 = nn.Linear(512, 256)
        nn.init.xavier_uniform_(self.fc_img_2.weight)

        self.fc_txt_1 = nn.Linear(TXT_LEN, 256)
        nn.init.xavier_uniform_(self.fc_txt_1.weight)

        self.fc_txt_2 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.fc_txt_2.weight)

        self.fc1 = nn.Linear(512, 128)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(128, 128)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.out = nn.Linear(128, N_CLASSES)
        nn.init.xavier_uniform_(self.out.weight)

        self.dropout = nn.modules.Dropout(p=0.25)

    def forward(self, inp_img, inp_txt):
        x_img = F.relu(self.fc_img_1(inp_img))
        x_img = self.dropout(x_img)
        x_img = F.relu(self.fc_img_2(x_img))
        x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt_1(inp_txt))
        x_txt = self.dropout(x_txt)
        x_txt = F.relu(self.fc_txt_2(x_txt))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class SelfAttentionModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.d = 256

        self.fc_img = nn.Linear(IMG_LEN, 128)
        self.fc_txt = nn.Linear(TXT_LEN, 128)

        self.fc_v = nn.Linear(self.d, self.d)
        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)

        self.fc_1 = nn.Linear(self.d, self.d)
        self.fc_2 = nn.Linear(self.d, self.d)
        self.out = nn.Linear(256, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=0.25)

    def forward(self, inp_img, inp_txt):

        m = inp_img.shape[0]

        x_img = F.relu(self.fc_img(inp_img))
        x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt(inp_txt))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), dim=1)

        v = self.fc_v(x)
        k = self.fc_k(x)
        q = self.fc_q(x)

        x_qk = torch.mm(q, torch.t(k)) / self.d ** (1./2)
        a = torch.nn.Softmax(dim=0)(torch.flatten(x_qk)).view(m, m)
        f = torch.mm(a, v)

        x = F.relu(self.fc_1(f))
        x = self.dropout(x)
        x = F.relu(self.fc_2(f))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class GSAHelper(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.d = d

        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)
        self.fc_kq = nn.Linear(self.d, self.d)

    def forward(self, k, q):
        m = k.shape[0]

        k_1 = self.fc_k(k)
        q_1 = self.fc_q(q)

        kq = nn.Sigmoid()(self.fc_kq(torch.mul(k_1, q_1)))

        k_2 = torch.mul(k, kq)
        q_2 = torch.mul(q, kq)

        mul = torch.mm(k_2, torch.t(q_2)) / self.d ** (1. / 2)
        a = nn.Softmax()(torch.flatten(mul)).view(m, m)
        return a


class GSA(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.d = d

        self.fc_v = nn.Linear(self.d, self.d)
        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)
        self.gsa_helper = GSAHelper(self.d)

    def forward(self, x):
        m = x.shape[0]

        v = self.fc_v(x)
        k = self.fc_k(x)
        q = self.fc_q(x)

        a = self.gsa_helper(k, q)
        f = torch.mm(a, v)
        return f


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.fc_1 = nn.Linear(2 * d, 4 * d)
        self.drop = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(4 * d, d)

    def forward(self, x_1, x_2):
        x = self.fc_1(torch.cat((x_1, x_2), 1))
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x


class UAModel1(nn.Module):
    def __init__(self, d=256):
        super().__init__()

        self.fc_img = nn.Linear(IMG_LEN, d // 2)
        self.fc_txt = nn.Linear(TXT_LEN, d // 2)

        self.d = d

        self.gsa_1 = GSA(d)
        self.ffn_1 = FFN(d)
        self.fc_out = nn.Linear(d, N_CLASSES)

    def forward(self, inp_img, inp_txt):
        x_img = self.fc_img(inp_img)
        x_txt = self.fc_txt(inp_txt)
        z = torch.cat((x_img, x_txt), 1)

        x = self.ffn_1(z, self.gsa_1(z))
        out = F.log_softmax(self.fc_out(x))
        return out


class UAModel2(nn.Module):
    def __init__(self, d=32):
        super().__init__()

        self.fc_img = nn.Linear(IMG_LEN, d // 2)
        self.fc_txt = nn.Linear(TXT_LEN, d // 2)

        self.d = d

        self.gsa_1 = GSA(d)
        self.ffn_1 = FFN(d)

        self.gsa_2 = GSA(d)
        self.ffn_2 = FFN(d)

        self.fc_out = nn.Linear(d, N_CLASSES)

    def forward(self, inp_img, inp_txt):
        x_img = self.fc_img(inp_img)
        x_txt = self.fc_txt(inp_txt)
        z = torch.cat((x_img, x_txt), 1)

        x = self.ffn_1(z, self.gsa_1(z))
        x = self.ffn_2(x, self.gsa_2(x))

        out = F.log_softmax(self.fc_out(x))
        return out


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


class ExtremeTrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(IMG_LEN + TXT_LEN, N_CLASSES)

    def forward(self, inp_img, inp_txt):
        x = torch.cat((inp_img, inp_txt), 1)
        x = F.log_softmax(self.fc(x), dim=1)
        return x
