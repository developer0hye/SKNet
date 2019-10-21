import torch
from torch import nn


class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):

        feas = []
        
        for conv in self.convs:
            fea = conv(x).unsqueeze_(dim=1)
            feas.append(fea)
        
        feas = torch.cat(feas, 1)
        
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        
        attention_vectors = []
        
        for fc in self.fcs:
            vector = fc(fea_z).unsqueeze_(dim=1)
            attention_vectors.append(vector)
        
        attention_vectors = torch.cat(attention_vectors, 1)
        
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class SKNet(nn.Module):
    def __init__(self, class_num):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 32x32
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(256, 256, 2, 8, 2),
            nn.ReLU(),
            SKUnit(256, 256, 2, 8, 2),
            nn.ReLU()
        ) # 32x32
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(512, 512, 2, 8, 2),
            nn.ReLU(),
            SKUnit(512, 512, 2, 8, 2),
            nn.ReLU()
        ) # 16x16
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(1024, 1024, 2, 8, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 2, 8, 2),
            nn.ReLU()
        ) # 8x8
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, class_num)


    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


if __name__=='__main__':
    x = torch.rand(8, 64, 32, 32)
    conv = SKConv(64, 3, 8, 2)
    out = conv(x)
    criterion = nn.L1Loss()
    loss = criterion(out, x)
    loss.backward()
    print('out shape : {}'.format(out.shape))
    print('loss value : {}'.format(loss))
