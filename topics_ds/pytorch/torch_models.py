import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


# implied that early stopping is only on train loss
def fit_topics_model(
    model, 
    optimizer, 
    train_loader, 
    val_loader=None, 
    scheduler=None, 
    writer=None, 
    epochs=1, 
    es_dif=None, 
    es_tol=0
):
    prev_train_loss = None
    tol_epochs = 0
    
    for epoch in range(epochs):
        model.train()

        train_loss_sum = 0.0
        train_loss_count = 0
        margin_sum = 0.0
        margin_count = 0

        for x_img_cur, x_txt_cur, y_cur in train_loader:
            model.zero_grad()
            output = model(x_img_cur, x_txt_cur)
            
            part = np.partition(-output.detach().numpy(), 1, axis=1)
            margin = - part[:, 0] + part[:, 1]
            margin = torch.tensor(margin.reshape(-1, 1)).float()
            margin_sum += sum(margin)
            margin_count += margin.shape[0]
            
            train_loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
            train_loss.backward()

            train_loss_sum += train_loss
            train_loss_count += 1

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        avg_train_loss = train_loss_sum / train_loss_count
        
        print('epoch:', epoch, 'train_loss:', train_loss, 'average train loss', avg_train_loss)
        print('margin_count', margin_count)
        print('average margin:', margin_sum / margin_count)
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('avg_train_loss', avg_train_loss, epoch)

        if val_loader is not None:
            model.eval()

            correct = 0
            total = 0
            val_loss_sum = 0.0
            val_loss_count = 0

            with torch.no_grad():
                for x_img_cur, x_txt_cur, y_cur in val_loader:
                    output = model(x_img_cur, x_txt_cur)
                    val_loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
                    val_loss_sum += val_loss
                    val_loss_count += 1
                    for idx, i in enumerate(output):
                        if torch.argmax(i) == torch.argmax(y_cur, dim=1)[idx]:
                            correct += 1
                        total += 1

            print('val_acc:', correct / total, 'val_avg_loss:', val_loss_sum / val_loss_count)
            if writer is not None:
                writer.add_scalar('val_acc', correct / total, epoch)
                writer.add_scalar('val_avg_loss', val_loss_sum / val_loss_count, epoch)
                
        # es part
        if es_dif is not None and epoch != 0:
            if tol_epochs != 0:  # already in tolerance mode
                if prev_train_loss - avg_train_loss > es_dif: # leave tolerance mode
                    tol_epochs = 0
                elif tol_epochs >= es_tol: # tolerance limit exceeded
                    return
                else: # continue tolerance mode
                    tol_epochs += 1
            elif prev_train_loss - avg_train_loss <= es_dif: # not in tolerance but to slow learning
                if es_tol == 0:  # no tolerance
                    return 
                else: #enter tolerance mode
                    tol_epochs += 1
        prev_train_loss = avg_train_loss
                
             
                
def fit_topics_trident_model(model, optimizer, train_loader, val_loader=None, scheduler=None, writer=None, epochs=1):
    for epoch in range(epochs):
        model.train()

        loss_sum = 0.0
        loss_common_sum = 0.0
        loss_img_sum = 0.0
        loss_txt_sum = 0.0
        loss_count = 0

        for x_img_cur, x_txt_cur, y_cur in train_loader:
            model.zero_grad()
            out_common, out_img, out_txt = model(x_img_cur, x_txt_cur)
            target = torch.argmax(y_cur, dim=1)
            loss_common = F.nll_loss(out_common, target)
            loss_img = F.nll_loss(out_img, target)
            loss_txt = F.nll_loss(out_txt, target)
            loss = (loss_common + loss_img + loss_txt) / 3.0
            loss.backward()

            loss_common_sum += loss_common
            loss_img_sum += loss_img
            loss_txt_sum += loss_txt
            loss_sum += loss
            loss_count += 1

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        print('epoch:', epoch, 'train_loss:', loss, 'average train loss', loss_sum / loss_count)
        print( 
            'avg common loss:', loss_common_sum / loss_count, 
            'avg img loss:', loss_img_sum / loss_count,
            'avg txt loss:', loss_txt_sum / loss_count
        )
        if writer is not None:
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('avg_train_loss', loss_sum / loss_count, epoch)
            writer.add_scalar('avg_train_loss_common', loss_common_sum / loss_count, epoch)
            writer.add_scalar('avg_train_loss_img', loss_img_sum / loss_count, epoch)
            writer.add_scalar('avg_train_loss_txt', loss_txt_sum / loss_count, epoch)



        if val_loader is not None:
            model.eval()

            correct_common = 0
            correct_img = 0
            correct_txt = 0
            total = 0
            loss_common_sum = 0.0
            loss_img_sum = 0.0
            loss_txt_sum = 0.0
            loss_sum = 0.0
            loss_count = 0

            with torch.no_grad():
                for x_img_cur, x_txt_cur, y_cur in val_loader:
                    out_common, out_img, out_txt = model(x_img_cur, x_txt_cur)
                    target = torch.argmax(y_cur, dim=1)
                    loss_common = F.nll_loss(out_common, target)
                    loss_img = F.nll_loss(out_img, target)
                    loss_txt = F.nll_loss(out_txt, target)
                    
                    loss = (loss_common + loss_img + loss_txt) / 3.0
                    
                    loss_common_sum += loss_common
                    loss_img_sum += loss_img
                    loss_txt_sum += loss_txt
                    loss_sum += loss
                    
                    loss_count += 1
                    for idx, i in enumerate(out_common):
                        if torch.argmax(i) == target[idx]:
                            correct_common += 1
                        total += 1
                    
                    for idx, i in enumerate(out_img):
                        if torch.argmax(i) == target[idx]:
                            correct_img += 1
                           
                    for idx, i in enumerate(out_txt):
                        if torch.argmax(i) == target[idx]:
                            correct_txt += 1
                    
            print(
                'val common acc:', correct_common / total,
                'val img acc:', correct_img / total,
                'val txt acc:', correct_txt / total,
                'val_avg_loss:', loss_sum / loss_count)
            print( 
                'avg common val loss:', loss_common_sum / loss_count, 
                'avg img val loss:', loss_img_sum / loss_count,
                'avg txt val loss:', loss_txt_sum / loss_count
            )
            if writer is not None:
                writer.add_scalar('val_acc', correct_common / total, epoch)
                writer.add_scalar('val_img_acc', correct_img / total, epoch)
                writer.add_scalar('val_txt_acc', correct_txt / total, epoch)
                writer.add_scalar('val_avg_loss', loss_sum / loss_count, epoch)
                

class NormModel(nn.Module):
    def __init__(self, d=128, drop=0.25, residual=False):
        super().__init__()
        self.residual = residual

        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)

        self.fc1 = nn.Linear(d * 4, d if not residual else d * 2)
        self.fc2 = nn.Linear(d if not residual else d * 6, d)
        self.out = nn.Linear(d, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

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
        x = F.relu(self.fc2(x if not self.residual else torch.cat((x_img, x_txt, x), 1)))

        x = F.log_softmax(self.out(x), dim=1)
        return x


class NormModelTrident(nn.Module):
    def __init__(self, d=128, drop=0.25, residual=False):
        super().__init__()
        self.residual = residual

        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)
        
        self.fc1 = nn.Linear(d * 4, d if not residual else d * 2)
        self.fc2 = nn.Linear(d if not residual else d * 6, d)
        self.out = nn.Linear(d, N_CLASSES)
        
        self.out_img = nn.Linear(d * 2, N_CLASSES)
        self.out_txt = nn.Linear(d * 2, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

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
        x = F.relu(self.fc2(x if not self.residual else torch.cat((x_img, x_txt, x), 1)))

        out = F.log_softmax(self.out(x), dim=1)
        out_img = F.log_softmax(self.out_img(x_img), dim=1)
        out_txt = F.log_softmax(self.out_txt(x_txt), dim=1)
        
        return out, out_img, out_txt

        
def fit_mtl(mtl, optimizer, epochs, train_loader, val_loader, writer):
    for _ in range(epochs):
        mtl.train()

        loss_sum = 0.0
        loss_count = 0

        for x_img_cur, x_txt_cur, y_cur in train_loader:
            mtl.zero_grad()
            loss = mtl(x_img_cur, x_txt_cur, y_cur)
            loss.backward()

            loss_sum += loss
            loss_count += 1

            optimizer.step()

        print('epoch:', epoch, 'train_loss:', loss, 'average train loss', loss_sum / loss_count)
        if writer is not None:
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('avg_train_loss', loss_sum / loss_count, epoch)

        if val_loader is not None:
            mtl.eval()

            correct = 0
            total = 0
            loss_sum = 0.0
            loss_count = 0

            with torch.no_grad():
                for x_img_cur, x_txt_cur, y_cur in val_loader:
                    output = mtl.model(x_img_cur, x_txt_cur)
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
        
        
class TridentMTL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigma = nn.Parameter(torch.ones(3))
        
    def forward(self, inp_img, inp_txt, target):
        outputs = self.model(inp_img, inp_txt)
        print(target.shape, output.shape)
        l = [F.nll_loss(output, target) for output in outputs]
        l = 0.5 * torch.Tensor(l) / self.sigma ** 2
        l = l.sum() + torch.log(self.sigma.prod())
        return l
    
    
class NormModelTridentBN(nn.Module):
    def __init__(self, d=128, drop=0.25):
        super().__init__()

        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.bn_img_1 = nn.BatchNorm1d(num_features=d*4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)
        self.bn_img_2 = nn.BatchNorm1d(num_features=d*2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.bn_txt_1 = nn.BatchNorm1d(num_features=d*2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)
        self.bn_txt_2 = nn.BatchNorm1d(num_features=d*2)
        
        self.fc1 = nn.Linear(d * 4, d)
        self.bn1 = nn.BatchNorm1d(num_features=d)

        self.fc2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(num_features=d)        
        
        self.out = nn.Linear(d, N_CLASSES)
        self.out_img = nn.Linear(d * 2, N_CLASSES)
        self.out_txt = nn.Linear(d * 2, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = self.bn_img_1(F.relu(self.fc_img_1(inp_img)))
        x_img = self.dropout(x_img)
        x_img = self.bn_img_2(F.relu(self.fc_img_2(x_img)))
        x_img = self.dropout(x_img)

        x_txt = self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt)))
        x_txt = self.dropout(x_txt)
        x_txt = self.bn_txt_2(F.relu(self.fc_txt_2(x_txt)))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))

        out = F.log_softmax(self.out(x), dim=1)
        out_img = F.log_softmax(self.out_img(x_img), dim=1)
        out_txt = F.log_softmax(self.out_txt(x_txt), dim=1)
        
        return out, out_img, out_txt
    
    
# class NormModelBN(nn.Module):
#     def __init__(self, d=128, drop=0.25):
#         super().__init__()

#         self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
#         self.bn_img_1 = nn.BatchNorm1d(num_features=d*4)
#         self.fc_img_2 = nn.Linear(d * 4, d * 2)
#         self.bn_img_2 = nn.BatchNorm1d(num_features=d*2)

#         self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
#         self.bn_txt_1 = nn.BatchNorm1d(num_features=d*2)
#         self.fc_txt_2 = nn.Linear(d * 2, d * 2)
#         self.bn_txt_2 = nn.BatchNorm1d(num_features=d*2)

#         self.fc1 = nn.Linear(d * 4, d)
#         self.bn1 = nn.BatchNorm1d(num_features=d)
#         self.fc2 = nn.Linear(d, d)
#         self.bn2 = nn.BatchNorm1d(num_features=d)
#         self.out = nn.Linear(d, N_CLASSES)

#         self.dropout = nn.modules.Dropout(p=drop)
        
#     def forward(self, inp_img, inp_txt):
#         x_img = F.relu(self.bn_img_1(self.fc_img_1(inp_img)))
#         x_img = self.dropout(x_img)
#         x_img = F.relu(self.bn_img_2(self.fc_img_2(x_img)))
#         x_img = self.dropout(x_img)

#         x_txt = F.relu(self.bn_txt_1(self.fc_txt_1(inp_txt)))
#         x_txt = self.dropout(x_txt)
#         x_txt = F.relu(self.bn_txt_2(self.fc_txt_2(x_txt)))
#         x_txt = self.dropout(x_txt)

#         x = torch.cat((x_img, x_txt), 1)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn2(self.fc2(x)))

#         x = F.log_softmax(self.out(x), dim=1)
#         return x

    
class NormModelBN2(nn.Module):
    def __init__(self, d=128, drop=0.25):
        super().__init__()

        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.bn_img_1 = nn.BatchNorm1d(num_features=d*4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)
        self.bn_img_2 = nn.BatchNorm1d(num_features=d*2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.bn_txt_1 = nn.BatchNorm1d(num_features=d*2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)
        self.bn_txt_2 = nn.BatchNorm1d(num_features=d*2)

        self.fc1 = nn.Linear(d * 4, d)
        self.bn1 = nn.BatchNorm1d(num_features=d)
        self.fc2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(num_features=d)
        self.out = nn.Linear(d, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)
        
    def forward(self, inp_img, inp_txt):
        x_img = self.bn_img_1(F.relu(self.fc_img_1(inp_img)))
        x_img = self.dropout(x_img)
        x_img = self.bn_img_2(F.relu(self.fc_img_2(x_img)))
        x_img = self.dropout(x_img)

        x_txt = self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt)))
        x_txt = self.dropout(x_txt)
        x_txt = self.bn_txt_2(F.relu(self.fc_txt_2(x_txt)))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))

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


class ResidualConcatModel(nn.Module):
    def __init__(self, d=128, drop=0.25):
        super().__init__()

        self.fc_img = nn.Linear(IMG_LEN, d)
        self.fc_txt = nn.Linear(TXT_LEN, d)

        self.fc1 = nn.Linear(2 * d, 2 * d)
        self.fc2 = nn.Linear(4 * d, 2 * d)
        self.out = nn.Linear(2 * d, N_CLASSES)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = F.relu(self.fc_img(inp_img))
        x_img = self.dropout(x_img)

        x_txt = F.relu(self.fc_txt(inp_txt))
        x_txt = self.dropout(x_txt)

        x = torch.cat((x_img, x_txt), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.cat((x_img, x_txt, x), 1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

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

        x_qk = torch.mm(q, torch.t(k)) / self.d ** (1. / 2)
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
    def __init__(self, d, drop=0.25):
        super().__init__()

        self.fc_1 = nn.Linear(2 * d, 4 * d)
        self.drop = nn.Dropout(drop)
        self.fc_2 = nn.Linear(4 * d, d)

    def forward(self, x_1, x_2):
        x = self.fc_1(torch.cat((x_1, x_2), 1))
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x


class UAModel1(nn.Module):
    def __init__(self, d=256, drop=0.25):
        super().__init__()

        self.fc_img = nn.Linear(IMG_LEN, d // 2)
        self.fc_txt = nn.Linear(TXT_LEN, d // 2)

        self.d = d

        self.gsa_1 = GSA(d)
        self.ffn_1 = FFN(d, drop=0.25)
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
