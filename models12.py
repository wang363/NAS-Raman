import copy
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from ops import ConvBNRelu, PRIMITIVES, Flatten

import torch
from torch import nn
from kanlayer import KAN
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SuperNetOutWeights(nn.Module):
    def __init__(self, in_channels):
        super(SuperNetOutWeights, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, 22, kernel_size=3, stride=1,padding=1),
                        nn.ReLU()
                        ) 
        self.convtranspose_500 = nn.Sequential(
                        nn.ConvTranspose1d(22, 22, kernel_size=5, stride=5,padding=0),
                        nn.ReLU()
                        )
         
        self.conv2 = nn.Sequential(nn.Conv1d(22, 110, kernel_size=1, stride=1,padding=0),
                                   nn.ReLU()
                                   )
        self.conv22_1 = nn.Conv1d(22,1,1,1,0)
        self.fc1 = nn.Linear(2200, 22)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim=1)  # nn.Sigmoid()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        batshsize = x.size(0)
        x_1 = self.conv1(x)
        weight = copy.copy(x_1)
        x_2 = self.convtranspose_500(x_1)
       
        result_x_1 = x_1.flatten(1)
        result_x_1 = self.fc1(result_x_1)
        result_x_1 = self.act_fn(result_x_1)
        result_x_1 = self.softmax(0.2*result_x_1)
        features = self.conv22_1(weight)#
        result = result_x_1 

        return result, weight
CANDIDATE_BLOCKS = ["trans", "graph","GSAU","ir_k3_e1","ir_k5_e1","ir_k7_e1","skip"]#1

class MixedOperation(nn.Module):
    def __init__(self, layer_parameters, proposed_operations):
        super(MixedOperation, self).__init__()
        self.ops_names = list(proposed_operations.keys())
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters) for op_name in self.ops_names])

        output_channels = layer_parameters[1]
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(output_channels) for _ in range(len(self.ops_names))])
        
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(self.ops_names) for _ in range(len(self.ops_names))]))

        self.dropout = nn.Dropout(p=0.1)
        self.attention = nn.Parameter(torch.Tensor(output_channels, len(self.ops_names)))
        nn.init.xavier_uniform_(self.attention)

        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, temperature):
        soft_mask_variables = F.gumbel_softmax(self.thetas, self.temperature)
        outputs = [norm(op(x)) for norm, op in zip(self.batch_norms, self.ops)]
        attn_weights = F.softmax(self.attention, dim=1)
        outputs = [m * out * attn_weights[:, i].unsqueeze(0).unsqueeze(2) 
                   for i, (m, out) in enumerate(zip(soft_mask_variables, outputs))]
        
        output = sum(outputs)
        output = self.dropout(output)
        return output
    
    def update_weights(self, feedback_scores):
        feedback_scores = feedback_scores.to(self.thetas.device)
        self.thetas.data = torch.softmax(self.thetas.data + feedback_scores, dim=0)
    
class SearchOperation(nn.Module):
        def __init__(self, SERCHSPACE, search,arc = [], i = 0):
            super(SearchOperation, self).__init__()
            self.SEARCH_SPACE = SERCHSPACE
            self.search = search
            if self.search:
                self.stages = None
                self.cnt_layers = np.size(self.SEARCH_SPACE["channel_size"])
                self.layers_parameters = [(self.SEARCH_SPACE["input_shape"][layer_id][0],        # C_in
                                self.SEARCH_SPACE["channel_size"][layer_id],               # C_out     
                                -999, # expansion (set to -999) embedded into operation and will not be considered
                                self.SEARCH_SPACE["strides"][layer_id],                    # stride
                                self.SEARCH_SPACE["input_shape"][layer_id][-1]              # length
                                ) for layer_id in range(self.cnt_layers)]

                self.operations = {op_name : PRIMITIVES[op_name] for op_name in CANDIDATE_BLOCKS}

                self.stages_to_search = nn.ModuleList([MixedOperation(
                                                    self.layers_parameters[layer_id],
                                                    self.operations)
                                                for layer_id in range(self.cnt_layers)]) 
            elif self.search == False:
                self.stages = nn.ModuleList([
                    PRIMITIVES[arc[i]](C_in=self.SEARCH_SPACE["input_shape"][0][0], 
                                                C_out=self.SEARCH_SPACE["channel_size"][0], 
                                                expansion=-1, 
                                                stride=self.SEARCH_SPACE["strides"][0], 
                                                length=self.SEARCH_SPACE["input_shape"][0][-1])
                ])
        def forward(self, x, temperature):
            if self.search:
            
                for mixed_op in self.stages_to_search:
                    y = mixed_op(x, temperature)
            else :
                for stage in self.stages:
                    y = stage(x)
            return y

class MultiKernalConv(nn.Module):
    def __init__(self, in_channels, out_channels, length) -> None:
        super().__init__()

        self.length = length
        n_feats = in_channels
        self.norm = nn.LayerNorm([n_feats, self.length])
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        self.K7 = nn.Sequential(
            nn.Conv1d(n_feats // 4, n_feats // 4, 7, 1, 7 // 2, groups=n_feats // 4),
            nn.Conv1d(n_feats // 4, n_feats // 4, 9, 1, (9 // 2) * 4, groups=n_feats // 4, dilation=4),
            nn.Conv1d(n_feats // 4, n_feats // 4, 1, 1, 0)
        )
        self.K5 = nn.Sequential(
            nn.Conv1d(n_feats // 4, n_feats // 4, 5, 1, 5 // 2, groups=n_feats // 4),
            nn.Conv1d(n_feats // 4, n_feats // 4, 7, 1, (7 // 2) * 3, groups=n_feats // 4, dilation=3),
            nn.Conv1d(n_feats // 4, n_feats // 4, 1, 1, 0)
        )
        self.K3 = nn.Sequential(
            nn.Conv1d(n_feats // 4, n_feats // 4, 3, 1, 1, groups=n_feats // 4),
            nn.Conv1d(n_feats // 4, n_feats // 4, 5, 1, (5 // 2) * 2, groups=n_feats // 4, dilation=2),
            nn.Conv1d(n_feats // 4, n_feats // 4, 1, 1, 0)
        )
        self.K1 = nn.Sequential(
            nn.Conv1d(n_feats // 4, n_feats // 4, 1, 1, 0, groups=n_feats // 4),
            nn.Conv1d(n_feats // 4, n_feats // 4, 3, 1, 1, groups=n_feats // 4),
            nn.Conv1d(n_feats // 4, n_feats // 4, 1, 1, 0)
        )

        self.conv3 = nn.Conv1d(n_feats // 4, n_feats // 4, 3, 1, 1, groups=n_feats // 4)
        self.conv5 = nn.Conv1d(n_feats // 4, n_feats // 4, 5, 1, 5 // 2, groups=n_feats // 4)
        self.conv7 = nn.Conv1d(n_feats // 4, n_feats // 4, 7, 1, 7 // 2, groups=n_feats // 4)
        self.conv1 = nn.Conv1d(n_feats // 4, n_feats // 4, 1, 1, 0, groups=n_feats // 4)
        self.inconv = nn.Conv1d(n_feats, n_feats * 2, 1)
        self.outconv = nn.Conv1d(n_feats, n_feats, 1)

    def forward(self, x) -> torch.Tensor:
        shortcut = x.clone()

        x = self.norm(x)
        x = self.inconv(x)
        a, x = torch.chunk(x, 2, dim=1)
        a1, a3, a5, a7 = torch.chunk(a, 4, dim=1)
        a = torch.cat([self.K1(a1) * self.conv1(a1), self.K3(a3) * self.conv3(a3), self.K5(a5) * self.conv5(a5), self.K7(a7) * self.conv7(a7)], dim=1)
        x = self.outconv(x * a)
        x = x * self.scale + shortcut
        return x

class StartModule(nn.Module):
    def __init__(self, inChannels, outChannels, search = True, arc = [] , i = 0):
        super(StartModule, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
            #### input shapes of all searched layers (considering with strides)
            ("input_shape", [(32, 200)]),
            # filter numbers over all layers
            ("channel_size", [32]),
            # strides over all layers
            ("strides", [1])
            ])
        self.search = search

       
        self.first = ConvBNRelu(input_depth=inChannels, output_depth=outChannels, kernel=3, stride=1, # 输入通道
                        pad=1, no_bias=1, use_relu="relu", bn_type="bn")
        
        self.rebult = MultiKernalConv(32, 32, 1000)
        self.sec = nn.Sequential(ConvBNRelu(input_depth=32, output_depth=32, kernel=3, stride=2, # 输入通道
                pad=1, no_bias=1, use_relu="relu", bn_type="bn"))
        self.avgpool = nn.AdaptiveAvgPool1d(200)

        self.searchOP = SearchOperation(self.SEARCH_SPACE, search,arc = arc, i = i)
    def forward(self, x, temperature):

        y = self.first(x)
        y = self.sec(y)
        y = self.avgpool(y)
        y = self.searchOP(y, temperature)
        
        return y 

class MidModule(nn.Module):
    def __init__(self, inChannels, outChannels, search = True, arc = [] , i = 0):
        super(MidModule, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
            #### input shapes of all searched layers (considering with strides)
            ("input_shape", [(32, 200)]),
            # filter numbers over all layers
            ("channel_size", [32]),
            # strides over all layers
            ("strides", [1])
            ])
        self.search = search
        self.searchOP = SearchOperation(self.SEARCH_SPACE, search,arc = arc, i = i)
        self.relu = nn.ReLU()
    def forward(self, x, temperature):

        y = self.searchOP(x, temperature)
        y = self.relu(y)

        return y  

class endModule(nn.Module):
    def __init__(self, inChannels, outChannels, search = True, arc = [], i = 0):
        super(endModule, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
            #### input shapes of all searched layers (considering with strides)
            ("input_shape", [(inChannels, 200)]),
            # filter numbers over all layers
            ("channel_size", [outChannels]),
            # strides over all layers
            ("strides", [5])
            ])
        self.search = search

        self.searchOP = SearchOperation(self.SEARCH_SPACE, search, arc = arc, i = i)
        self.relu = nn.ReLU()
    def forward(self, x, temperature):
        y = self.searchOP(x, temperature)
        y = self.relu(y)
        return y

class Down(nn.Module):
    '''Downsampling module'''
    def __init__(self, shape, outChannel, stride=2,search = True, arc = None, i =0):
        super(Down, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
        #### input shapes of all searched layers (considering with strides)
        ("input_shape", [( shape[0] , (shape[-1]//stride))]),
        # filter numbers over all layers
        ("channel_size", [outChannel]),
        # strides over all layers
        ("strides", [1])
        ])

        self.search = search
        self.searchOP = SearchOperation(self.SEARCH_SPACE, search, arc = arc, i = i)
        self.maxpool = nn.MaxPool1d(stride, stride=stride)

        self.relu = nn.ReLU()
    def forward(self, x, temperature):
        y = self.maxpool(x)
        y = self.searchOP(y, temperature)
        y = self.relu(y)
        return y
    
class Up(nn.Module):
    def __init__(self, shape, outChannel, bilinear=False, stride=1, search = True,arc = [], i = 0):
        super(Up, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
        #### input shapes of all searched layers (considering with strides)
        ("input_shape", [( shape[0]*2, shape[-1])]),
        # filter numbers over all layers
        ("channel_size", [outChannel]),
        # strides over all layers
        ("strides", [1])
            ])

        self.search = search
        self.searchOP = SearchOperation(self.SEARCH_SPACE, search, arc = arc, i = i)
       
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(shape[0]*2, shape[0], kernel_size=2, stride=2, padding=0) # 降维

        self.relu = nn.ReLU()
    def forward(self, x1,x2,temperature):
        x1 = self.up(x1)
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])    
        x = torch.cat([x2, x1], dim=1)
        y = self.searchOP(x, temperature) # 此处channel降维
        y = self.relu(y)
        return y



class SuperNet(nn.Module):
    def __init__(self,cnt_classes= 22):
        super(SuperNet, self).__init__()
        self.inc = StartModule(1, 32)
        self.mid_1 = MidModule(32,32)
        self.mid_2 = MidModule(32,32)
        self.down1 = Down(( 32, 200), 64,stride=2)
        self.down2 = Down(( 64, 100), 128,stride=2)
        self.up1 = Up(( 64, 100), 64, stride=2)
        self.up2 = Up(( 32, 200), 32, stride= 2)

        self.down3 = Down(( 32, 200), 64,stride=2)
        self.down4 = Down(( 64, 100), 128,stride=2)
        self.up3 = Up(( 64, 100), 64, stride=2)
        self.up4 = Up(( 32, 200), 32, stride= 2)

        self.outc = endModule(64, cnt_classes)

        self.sig = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool1d(100)
        self.last_stages = SuperNetOutWeights(22)
        self.layers_hidden = [880, 88, cnt_classes]  
        self.kan_module = nn.Sequential(OrderedDict([
            ("dropout", nn.Dropout(p=0.1)),
            ("flatten", Flatten()),
            ("linear",nn.Linear(880, 22)),
            # ("kan",KAN(self.layers_hidden)),
            ("softmax",nn.Sigmoid())
            ]))

    def forward(self, x, temperature):
        x = x.unsqueeze(1)
        x1  = self.inc(x, temperature) # 32, 200
        #获取 x1 的shape
        x1 = self.mid_1(x1, temperature)

        x2 = self.down1(x1, temperature) # 32, 100
        x3 = self.down2(x2, temperature) # 64, 50

        x = self.up1(x3, x2, temperature)
        x = self.up2(x, x1, temperature)

        x_2 = self.down3(x, temperature) # 32, 100
        x_3 = self.down4(x_2, temperature) # 64, 50
        x_1 = self.up3(x_3, x_2, temperature)
        x_1 = self.up4(x_1, x, temperature)

        x = self.mid_2(x_1,temperature)
        x = x.squeeze(2)
        y = torch.cat((x1,x),dim =1)
        y = self.outc(y, temperature)

        logits_1 = self.kan_module(y)
        

        y = self.avg(y)
        logits, Weights = self.last_stages(y)
        logits = 0.9 * logits_1 + 0.1 * logits
        # logits = logits_1
        # if self.output_features:
        #     return logits, Weights
        # else:

        return logits, Weights
    def update_weights(self, feedback_scores):
        # Iterate over all layers and update the weights of MixedOperation layers
        for module in self.modules():
            if isinstance(module, MixedOperation):
                module.update_weights(feedback_scores)
class SupernetLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SupernetLoss, self).__init__()
        # 权重参数
        self.alpha = alpha
        self.beta = beta
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor([0.0405, 0.0424, 0.0546, 0.0474, 0.0413, 0.0636, 0.0538, 0.0535, 0.0559,
        0.0357, 0.0368, 0.0354, 0.0346, 0.0482, 0.0571, 0.0525, 0.0383, 0.0559,
        0.0473, 0.0373, 0.0305, 0.0373],dtype=torch.float32,device=device))

    def weight_criterion(self, predictions, targets):
        predictions = predictions.float()
        targets = targets.float()
        tar = torch.tensor([1],dtype=torch.float).to(device)
        cosine_sim = self.cosine_loss(predictions,targets,tar)
        # 计算均方误差损失
        mse_loss = self.mse_loss(predictions, targets)
        combined_loss = (self.alpha * mse_loss) + (self.beta * cosine_sim)
        return combined_loss
    def forward(self, predictions, targets, losses_ce, N):
        loss = self.weight_criterion(predictions, targets)
        losses_ce.update(loss.item(), N)

        return loss

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class FBNet(nn.Module):
    def __init__(self, n_classes, arch_operations, output_features=False):
        super(FBNet, self).__init__()

        self.arch_operations = arch_operations
        self.output_features = output_features
        self.inc = StartModule(1, 32,search=False, arc=arch_operations, i=0)
        self.mid_1 = MidModule(32,32,search=False, arc=arch_operations, i=1)

        self.down1 = Down(( 32, 200), 64, search=False, arc=arch_operations, i=2)
        self.down2 = Down(( 64, 100), 128, search=False, arc=arch_operations, i=3)

        self.up1 = Up((64, 100), 64, stride=2, search=False, arc=arch_operations, i=4)
        self.up2 = Up((32, 200), 32, stride=2, search=False, arc=arch_operations, i=5)


        self.down3 = Down(( 32, 200), 64, search=False, arc=arch_operations, i=6)
        self.down4 = Down(( 64, 100), 128, search=False, arc=arch_operations, i=7)

        self.up3 = Up((64, 100), 64, stride=2, search=False, arc=arch_operations, i=8)
        self.up4 = Up((32, 200), 32, stride=2, search=False, arc=arch_operations, i=9)

        self.mid_2 = MidModule(32,32,search=False, arc=arch_operations, i=10)

        self.outc = endModule(64, n_classes, search=False, arc=arch_operations, i=11)
        self.avg = nn.AdaptiveAvgPool1d(100)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.last_stages = SuperNetOutWeights(n_classes)
        self.layers_hidden = [880,88,44, n_classes]  
        self.kan_module = nn.Sequential(OrderedDict([
            ("dropout", nn.Dropout(p=0.1)),
            ("flatten", Flatten()),

            # ("linear",nn.Linear(880, 88)),
            # ("relu", nn.ReLU()),
            ("linear2",nn.Linear(880, n_classes)),
            # ("kan",KAN(self.layers_hidden)),
            ("softmax",nn.Sigmoid())
            ]))

# 
    def forward(self, x, temperature):
        x = x.unsqueeze(1)
        x1  = self.inc(x, temperature)
        #获取 x1 的shape
        x1 = self.mid_1(x1, temperature)

        x2 = self.down1(x1, temperature) # 32, 100
        x3 = self.down2(x2, temperature) # 64, 50

        x = self.up1(x3, x2, temperature)
        x = self.up2(x, x1, temperature)

        x_2 = self.down3(x, temperature) # 32, 100
        x_3 = self.down4(x_2, temperature) # 64, 50
        x_1 = self.up3(x_3, x_2, temperature)
        x_1 = self.up4(x_1, x, temperature)

        x = self.mid_2(x_1,temperature)

        x = x.squeeze(2)
        y = torch.cat((x1,x),dim =1)
        y = self.outc(y,temperature)


        logits_1 = self.kan_module(y)
        
        y = self.avg(y) # 
        logits, Weights = self.last_stages(y)
        logits = 0.9 * logits_1 + 0.1 * logits
        # logits = logits_1
        # if self.output_features:
        #     return logits, Weights
        # else:

        return logits, Weights.transpose(-1,-2)
# arch_operations =  ['trans', 'skip', 'ir_k5_e1', 'skip', 'graph', 'ir_k7_e1', 'ir_k7_e1', 'trans', 'trans']       
img = torch.randn(4,1000)
model = SuperNet()
preds,_ = model(img, 0.1)
print(preds.shape)

# model = FBNet(arch_operations=arch_operations)
# preds = model(img, 0.1)
# print(preds.shape)
