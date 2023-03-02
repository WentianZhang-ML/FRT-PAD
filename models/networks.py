import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from models.layers import IBasicBlock, ResidualBlock
from models.layers import GraphAttentionLayer

class Baseline(nn.Module):
    """
    PAD extractor using ResNet 18
    """
    def __init__(self):
        super(Baseline, self).__init__()
        model_resnet = ResNet(BasicBlock, [2, 2, 2, 2])
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        
    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        out = self.avgpool(feature) #[,512,1,1]
        out = out.view(out.size(0), -1) #[,512]
        return out

class Face_Recognition(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block=IBasicBlock, layers=[2,2,2,2], dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(Face_Recognition, self).__init__()

        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None: 
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            feature_1 = self.layer1(x)
            feature_2 = self.layer2(feature_1)
            feature_3 = self.layer3(feature_2)
            feature_4 = self.layer4(feature_3)
            x = self.bn2(feature_4)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        out = self.features(x)
        return (feature_1, feature_2, feature_3, feature_4, out)
class Face_Expression(nn.Module):
    def __init__(self):
        super(Face_Expression, self).__init__()
        resnet  = models.resnet18()
        self.features = nn.Sequential(*list(resnet.children())[:-6])
        self.layer1 = nn.Sequential(*list(resnet.children())[-6:-5])
        self.layer2 = nn.Sequential(*list(resnet.children())[-5:-4])
        self.layer3 = nn.Sequential(*list(resnet.children())[-4:-3])
        self.layer4 = nn.Sequential(*list(resnet.children())[-3:-2]) 
        self.avgpool = list(resnet.children())[-2] 

    def forward(self, x):
        x = self.features(x)
        feature_1 = self.layer1(x)
        feature_2 = self.layer2(feature_1)
        feature_3 = self.layer3(feature_2)
        feature_4 = self.layer4(feature_3)
        out = self.avgpool(feature_4)
        out = out.view(out.size(0), -1)
        return  (feature_1, feature_2, feature_3, feature_4, out)
    
class Face_Attribute_D(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Face_Attribute_D, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        
        self.main1 = nn.Sequential(*layers[:2])
        self.main2 = nn.Sequential(*layers[2:4])
        self.main3 = nn.Sequential(*layers[4:6])
        self.main4 = nn.Sequential(*layers[6:8])
        self.main5 = nn.Sequential(*layers[8:10])
        self.main6 = nn.Sequential(*layers[10:12])

        self.fc1 = nn.Linear(1024*4*4, 512) # 1024*4*4 
        self.fc2 = nn.Linear(2048*2*2, 512) # 2048*2*2
        self.fc3 = nn.Linear(1024, 512)
        
    def forward(self, x):
        feature_1 = self.main1(x)
        feature_2 = self.main2(feature_1)
        feature_3 = self.main3(feature_2)
        feature_4 = self.main4(feature_3) # 512
        feature_5 = self.main5(feature_4) # 1024
        feature_6 = self.main6(feature_5) # 2048

        f5 = feature_5.view(feature_5.size(0), -1)
        f6 = feature_6.view(feature_6.size(0), -1)
        out = torch.cat([self.fc1(f5), self.fc2(f6)], 1)
        out = self.fc3(out)
        return (feature_1,feature_2,feature_3,feature_4,out)

class GAT(nn.Module):
    def __init__(self, adj, batch_size, nfeat=512, nhid=512, nclass=512, dropout=0.6, alpha=0.2, nheads=2):
        super(GAT, self).__init__()

        # Convolution extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,groups=1,bias=False), 
            nn.BatchNorm2d(128, eps=1e-05),
            nn.PReLU(128), 
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,groups=1,bias=False), 
            nn.BatchNorm2d(256, eps=1e-05),
            nn.PReLU(256),
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,groups=1,bias=False), 
            nn.BatchNorm2d(512, eps=1e-05),
            nn.PReLU(512),
            nn.AdaptiveAvgPool2d(output_size=1)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,groups=1,bias=False), 
            nn.BatchNorm2d(256, eps=1e-05),
            nn.PReLU(256), 
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,groups=1,bias=False), 
            nn.BatchNorm2d(512, eps=1e-05),
            nn.PReLU(512),
            nn.AdaptiveAvgPool2d(output_size=1)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,groups=1,bias=False), 
            nn.BatchNorm2d(512, eps=1e-05),
            nn.PReLU(512),
            nn.AdaptiveAvgPool2d(output_size=1)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1,groups=1,bias=False), 
            nn.BatchNorm2d(512, eps=1e-05),
            nn.PReLU(512),
            nn.AdaptiveAvgPool2d(output_size=1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
                # nn.init.xavier_normal(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.adj = adj
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat*batch_size, nhid*batch_size, dropout, alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads * batch_size, nclass * batch_size, dropout=dropout, alpha=alpha, concat=False)
        self.nclass = nclass
        self.batch_size = batch_size

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # self.fc = nn.Linear(nclass*5, nclass)

    def forward(self, x):
        f1 = self.conv1(x[0])
        f2 = self.conv2(x[1])
        f3 = self.conv3(x[2])
        f4 = self.conv4(x[3])
        f5 = x[4]
        f1 = torch.reshape(f1,[1,f1.shape[0]*f1.shape[1]])
        f2 = torch.reshape(f2,[1,f2.shape[0]*f2.shape[1]])
        f3 = torch.reshape(f3,[1,f3.shape[0]*f3.shape[1]])
        f4 = torch.reshape(f4,[1,f4.shape[0]*f4.shape[1]])
        f5 = torch.reshape(f5,[1,f5.shape[0]*f5.shape[1]])
        feature = torch.cat([f1,f2,f3,f4,f5],dim=0)        # (5,batch_size*512) 5 vertex
        #GAT
        x = F.dropout(feature, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))

        x1 = torch.reshape(x[0],[self.batch_size,self.nclass])
        x2 = torch.reshape(x[1],[self.batch_size,self.nclass])
        x3 = torch.reshape(x[2],[self.batch_size,self.nclass])
        x4 = torch.reshape(x[3],[self.batch_size,self.nclass])
        x5 = torch.reshape(x[4],[self.batch_size,self.nclass])   #(batch_size,512)

        
        features = torch.cat([x1,x2,x3,x4],dim = 1)
        features = self.avgpool(features)
        out = torch.mul(features,x5)
        return out

class PAD_Classifier(nn.Module):
    def __init__(self, PAE_net, downstream_net, target_net,downstream_name='FR'):
        super(PAD_Classifier, self).__init__()
        self.downstream_name = downstream_name
        self.classifier_layer = nn.Linear(1024, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.dropout = nn.Dropout(0.5)

        self.Extractor = PAE_net
        self.downstream = downstream_net
        self.targetnet = target_net

    def forward(self, x ):
        PAE_feature = self.Extractor(x)

        if self.downstream_name =='FR':
            x_ = F.interpolate(x,(112,112),mode='bilinear', align_corners=True)
        elif self.downstream_name == 'FE':
            x_ = F.interpolate(x,(224,224),mode='bilinear', align_corners=True)
        elif self.downstream_name =='FA':
            x_ = F.interpolate(x,(128,128),mode='bilinear', align_corners=True)

        downstream_features = self.downstream(x_)
        downstream_features_detach = [i.detach() for i in downstream_features]

        target_net_feature = self.targetnet(downstream_features_detach)

        feature_f = torch.cat([PAE_feature,target_net_feature],dim=1)
        self.dropout(feature_f)
        out = self.classifier_layer(feature_f)
        return out




