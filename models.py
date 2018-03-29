import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, cell_num=2):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.rnn = nn.GRU(input_size, hidden_size, cell_num, batch_first=True, dropout = 0.3)
    def forward(self, feats):
        h0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)

        if feats.is_cuda:
            h0 = h0.cuda()

        # aggregated feature
        feat, _ = self.rnn(feats, h0)
        return feat[:,-1,:]


# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()

        #CNN
        #self.convs = CONV(input_size=4,kernel_size=(5,5))
        
        #LSTM
        self.VFeatPool = FeatAggregate(1024, 128)
        self.AFeatPool = FeatAggregate(128, 128)
        self.fc = nn.Linear(7*128, 128*4)
        self.init_params()
        
        #self.dropout = nn.Dropout(0.5)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        
        
        #LSTM
        lgh = 30;
        step = 15;
        
        feat_v0 = vfeat[:,0:lgh,:]
        feat_v0 = self.VFeatPool(feat_v0)
        
        feat_a0 = afeat[:,0:lgh,:]
        feat_a0 = self.AFeatPool(feat_a0)
        
        l = (120-lgh)//step+1;

        lstm_v = Variable(torch.zeros(feat_v0.size(0),128*l))
        lstm_a = Variable(torch.zeros(feat_a0.size(0),128*l))
        
        if vfeat.is_cuda:
            lstm_v = lstm_v.cuda()
            lstm_a = lstm_a.cuda()
        
        lstm_v[:,0:128] = feat_v0
        lstm_a[:,0:128] = feat_a0
        
        count = 1
        
        for i in range(step,120-lgh+step,step):
            
            feat_v0 = vfeat[:,i:i+lgh,:]
            lstm_v[:,count*128:(count+1)*128] = self.VFeatPool(feat_v0)

            
            feat_a0 = afeat[:,i:i+lgh,:]
            lstm_a[:,count*128:(count+1)*128] = self.AFeatPool(feat_a0)
            
            count = count + 1
        
        lstm_v = self.fc(lstm_v)
        lstm_a = self.fc(lstm_a)
        
        '''
        cnnv = self.convs(lstm_v)  
        cnna = self.convs(lstm_a)
        cnnv = self.fc(cnnv)
        cnna = self.fc(cnna)
        '''
        
        return lstm_v,lstm_a


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=0.9):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor,positive,negative)
