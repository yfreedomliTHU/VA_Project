import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

#def Cosdist(x,y):
    
class BasicConv2d(nn.Module):
    """BasicConv2d model."""

    def __init__(self, in_channels, out_channels, **kwargs):
        """Init BasicConv2d model."""
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        """Forward BasicConv2d model."""
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, cell_num=2):
        super(FeatAggregate,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.rnn = nn.LSTM(input_size, hidden_size, cell_num, batch_first=True)
        self.fc = nn.Linear(4*hidden_size,256)

    def forward(self, feats):
        h0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        c0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)

        if feats.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # aggregated feature
        feat, _ = self.rnn(feats, (h0, c0))
        feat = feat[:,-5:-1,:]
        feat = feat.contiguous()
        feat = feat.view(feat.size()[0],-1)
        feat = self.fc(feat)
        return feat

class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric,self).__init__()
        self.conv_v1 = BasicConv2d(1,4,kernel_size=(3,3),stride=1,padding=1)
        self.conv_a1 = BasicConv2d(1,4,kernel_size=(3,1),stride=1,padding=(1,0))
        self.pool_v1 = nn.MaxPool2d((2,2),stride=(1,2))
        self.pool_a1 = nn.MaxPool2d((2,2),stride=(1,2))
        self.conv_v2 = BasicConv2d(4,8,kernel_size=3,stride=1,padding=1)
        self.conv_a2 = BasicConv2d(4,8,kernel_size=3,stride=1,padding=1)
        self.pool_v2 = nn.MaxPool2d(2,stride=2)
        self.pool_a2 = nn.MaxPool2d(2,stride=2)
        self.conv_v3 = BasicConv2d(8,8,kernel_size=3,stride=1,padding=1)
        self.pool_v3 = nn.MaxPool2d(2,stride=(1,2))
        self.rnn_v = FeatAggregate(1024,128)
        self.rnn_a = FeatAggregate(256,128)
        self.fc = nn.Linear(256,128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,vfeat,afeat):
        vfeat = vfeat.view(vfeat.size(0),1,vfeat.size(1),vfeat.size(2))
        afeat = afeat.view(afeat.size(0),1,afeat.size(1),afeat.size(2))
        # 1x120x1024
        vfeat = self.conv_v1(vfeat)
        # 16x120x1024
        vfeat = self.pool_v1(vfeat)
        # 16x119x512
        vfeat = self.conv_v2(vfeat)
        # 32x119x512
        vfeat = self.pool_v2(vfeat)
        # 32x59x256
        vfeat = self.conv_v3(vfeat)
        vfeat = self.pool_v3(vfeat)
        # 32x58x128
        vfeat = vfeat.transpose(1,2)
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0),vfeat.size(1),-1)
        vfeat = self.rnn_v(vfeat)
        vfeat = F.dropout(vfeat,0.4)
        vfeat = self.fc(vfeat)
        vfeat = F.relu(vfeat)
        #vfeat = F.softmax(vfeat)

        afeat = self.conv_a1(afeat)
        afeat = self.pool_a1(afeat)
        afeat = self.conv_a2(afeat)
        afeat = self.pool_a2(afeat)
        afeat = afeat.transpose(1,2)
        afeat = afeat.contiguous()
        afeat = afeat.view(afeat.size(0),afeat.size(1),-1)
        afeat = self.rnn_a(afeat)
        afeat = F.dropout(afeat,0.3)
        afeat = self.fc(afeat)
        afeat = F.relu(afeat)
        #afeat = F.softmax(afeat)

        out = torch.clamp(1-F.cosine_similarity(vfeat, afeat),max=1)
        return out





# Visual-audio multimodal metric learning: MaxPool+FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = vfeat.transpose(2, 1)
        vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 1024)
        vfeat = F.relu(self.vfc(vfeat))
        vfeat = self.fc(vfeat)

        # aggregate the auditory features
        afeat = afeat.transpose(2, 1)
        afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = self.fc(afeat)
        return F.pairwise_distance(vfeat, afeat)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss
