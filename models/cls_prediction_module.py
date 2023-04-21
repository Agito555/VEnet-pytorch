import torch
import torch.nn as nn
import torch.nn.functional as F

class Cls_pre(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            pred_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, 1 , 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        

    def forward(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            probabilty of seed points to be foreground: (batch_size,num_seed)
        """

        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # B, 1 ,num_seed 
        # output=F.sigmoid(net)

        output=net.transpose(2,1)

        # print(output.shape)
        # print(output)

        output=torch.squeeze(output) # B,num_seed

        # print(output.shape)
        # print(output)

        return output

if __name__=='__main__':
    net = Cls_pre(1, 256).cuda()
    prob = net(torch.rand(8,256,1024).cuda())



