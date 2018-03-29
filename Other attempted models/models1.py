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
        #return x
        return F.relu(x, inplace=True)


class CNN(nn.Module):
    def __init__(self,feature_size):
        super(CNN,self).__init__()
        self.feature_size = feature_size
        self.conv1 = BasicConv2d(1,feature_size,kernel_size=[1,feature_size])
        #self.conv2 = nn.Conv2d(1,feature_size,kernel_size=[1,2*feature_size])
        #self.conv3 = nn.Conv2d(1,feature_size,kernel_size=[1,3*feature_size])
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,feat):
        feat = feat.view(feat.size(0),1,120,-1)
        #feat1 = F.pad(feat,(0,0,1,0))[:,:,:120,:]
        #feat2 = F.pad(feat,(0,0,2,0))[:,:,:120,:]
        #feat1 = torch.cat((feat,feat1),3)
        #feat2 = torch.cat((feat1,feat2),3)

        #feat = feat.view(feat.size(0),1,120,-1)
        feat = self.conv1(feat)
        feat = feat.view(feat.size(0),120,-1)
        #feat1 = feat1.view(feat1.size(0),1,120,-1)
        #feat1 = self.conv2(feat1)
        #feat1 = feat1.view(feat1.size(0),120,-1)
        #feat2 = feat2.view(feat2.size(0),1,120,-1)
        #feat2 = self.conv3(feat2)
        #feat2 = feat.view(feat2.size(0),120,-1)

        #out = torch.cat((feat,feat1,feat2),2)
        return feat


class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128,framenum=120,cell_num=2):
        super(FeatAggregate,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.framenum = framenum
        self.rnn1 = nn.GRU(input_size, hidden_size, cell_num, batch_first=True)
        #self.rnn2 = nn.LSTM(hidden_size, hidden_size, cell_num, batch_first=True)
        #self.bp = nn.BatchNorm1d(hidden_size)
        #self.ap = nn.AvgPool1d(120)
        self.conv = nn.Conv1d(hidden_size,hidden_size*6,kernel_size=framenum)


    def forward(self, feats):
        h01 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        #c01 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        #h02 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        #c02 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)

        if feats.is_cuda:
            h01 = h01.cuda()
            #c01 = c01.cuda()
            #h02 = h02.cuda()
            #c02 = c02.cuda()

        # aggregated feature
        feat, hn1 = self.rnn1(feats, h01)
        #feat = torch.transpose(feat,1,2)
        #feat = feat.contiguous()
        #feat = self.bp(feat)
        #feat = torch.transpose(feat,1,2)
        #feat, (hn2, cn2) = self.rnn2(feat, (h02, c02))
        feat = feat.transpose(1,2)
        #feat = self.ap(feat)
        feat = self.conv(feat)
        #feat = feat[:,-1,:]
        feat = feat.view(feat.size(0),-1)
        #feat = feat.contiguous()
        out = torch.cat((feat,hn1[1],hn1[0]),1)
        
        return out

class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric,self).__init__()
        #self.pool_v = nn.AvgPool1d(2,stride=2)
        #self.pool_a = nn.AvgPool1d(2,stride=2)
        #self.rnn_vp = FeatAggregate(1024,128,60)
        #self.rnn_ap = FeatAggregate(128,128,60)
        #self.fc_vp = nn.Linear(2*128,2*128)
        #self.fc_ap = nn.Linear(2*128,2*128)
        self.rnn_v = FeatAggregate(1024,128,120)
        self.rnn_a = FeatAggregate(128,128,120)
        self.fc_v1 = nn.Linear(128*8,4*128)
        self.fc_v2 = nn.Linear(128*4,128)
        self.fc_a1 = nn.Linear(128*8,128*4)
        self.fc_a2 = nn.Linear(4*128,128)
        #self.fc = nn.Linear(4*128,128)
        #self.cnn_a = CNN(128)
        #self.cnn_v = CNN(1024)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)


    def forward(self,vfeat,afeat):
        
        
        
        #vfeat1 = F.sigmoid(vfeat1)

        #vfeat = self.cnn_v(vfeat)

        #vfeat1 = torch.transpose(vfeat,1,2)

        #vfeat1 = self.pool_v(vfeat1)
        #vfeat1 = torch.transpose(vfeat1,1,2)
        #vfeat1 = self.rnn_vp(vfeat1)
        #vfeat1 = self.fc_vp(vfeat1)
        #vfeat1 = F.normalize(vfeat1)
        vfeat = self.rnn_v(vfeat)
        #v = torch.cat((vfeat,vfeat1),1)
        #vfeat = F.relu(vfeat)
        #vfeat = torch.cat((vfeat,vfeat1),1)
        v = self.fc_v1(vfeat)
        v = F.sigmoid(v)
        v = F.dropout(v,0.0)
        v = self.fc_v2(v)
        #vfeat = F.normalize(vfeat)
        
        #vfeat = F.sigmoid(vfeat)
        #vfeat = F.dropout(vfeat,0.2)
        #vfeat = self.fc_v2(vfeat)
        #vfeat = F.sigmoid(vfeat)

        #afeat = self.cnn_a(afeat)
        
        #afeat1 = F.sigmoid(afeat1)
        

        #afeat = self.cnn_a(afeat)
        #afeat1 = torch.transpose(afeat,1,2)
        
        #afeat1 = self.pool_a(afeat1)
        #afeat1 = torch.transpose(afeat1,1,2)
        #afeat1 = self.rnn_ap(afeat1)
        #afeat1 = self.fc_ap(afeat1)
        #afeat1 = F.normalize(afeat1)
        afeat = self.rnn_a(afeat)
        
        #afeat = F.relu(afeat)
        #afeat = torch.cat((afeat,afeat1),1)
        #a = torch.cat((afeat,afeat1),1)
        a = self.fc_a1(afeat)
        a = F.sigmoid(a)
        a = F.dropout(a,0.0)
        a = self.fc_a2(a)

        #afeat = F.normalize(afeat)
        
        #afeat = F.dropout(afeat,0.0)
        #afeat = self.fc_a2(afeat)

        #afeat = F.softmax(afeat)
        #out = F.pairwise_distance(a,v)
        
        #out = torch.clamp(out,max=1.0)
        #out = (1-F.cosine_similarity(vfeat, afeat))/2
        return v,a





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
