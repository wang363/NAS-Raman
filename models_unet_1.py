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

operations_index =  [0,1,2,3,4,6] # Module Index
CANDIDATE_BLOCKS = ["trans", "graph","GSAU","ir_k3_e1","ir_k5_e1","ir_k7_e1","skip"] # Search Space
SEARCH_SPACE = OrderedDict([
    #### input shapes of all searched layers (considering with strides)
    ("input_shape", [(128, 200), (128,  200),  (128,  200), (128,  200), (128,  200), (128, 200)]),
    # filter numbers over all layers
    ("channel_size", [128,  128,  128, 128, 128,  128]),
    # strides over all layers
    ("strides", [1,  1,  1, 1, 1, 1])
            ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SuperNetOutWeights(nn.Module):
    def __init__(self, in_channels):
        super(SuperNetOutWeights, self).__init__()


        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, 22, kernel_size=3, stride=2,padding=1),
                        nn.ReLU()
                        ) # reshape成（44，50）-> (100,22)
        self.convtranspose_500 = nn.Sequential(
                        nn.ConvTranspose1d(22, 22, kernel_size=5, stride=5,padding=0),
                        nn.ReLU()
                        )
    

         
        self.conv2 = nn.Sequential(nn.Conv1d(22, 110, kernel_size=1, stride=1,padding=0),
                                   nn.ReLU()
                                   )
        self.fc1 = nn.Linear(2200, 22)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim=1)  # nn.Sigmoid()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        batshsize = x.size(0)
        x_1 = self.conv1(x)
        weight = copy.copy(x_1)#
        x_2 = self.convtranspose_500(x_1)
        result_x_1 = x_1.flatten(1)
        result_x_1 = self.fc1(result_x_1)
        result_x_1 = self.act_fn(result_x_1)
        result_x_1 = self.softmax(result_x_1)
        result = result_x_1 

        return result, weight

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
        def __init__(self, SEARCH_SPACE=SEARCH_SPACE, search=None,arc = [], i = 0):
            super(SearchOperation, self).__init__()
            self.SEARCH_SPACE = SEARCH_SPACE
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
    
class endModule(nn.Module):
    def __init__(self, inChannels, outChannels, search = True, arc = [], i = 0):
        super(endModule, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
            #### input shapes of all searched layers (considering with strides)
            ("input_shape", [(inChannels, 200)]),
            # filter numbers over all layers
            ("channel_size", [outChannels]),
            # strides over all layers
            ("strides", [1])
            ])
        self.search = search

        self.searchOP = SearchOperation(self.SEARCH_SPACE, search, arc = arc, i = i)
        self.relu = nn.ReLU()
    def forward(self, x, temperature):
        y = self.searchOP(x, temperature)
        y = self.relu(y)
        return y

class StartModule(nn.Module):
    def __init__(self, search = True, arc = [] , i = 0):
        super(StartModule, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
            #### input shapes of all searched layers (considering with strides)
            ("input_shape", [(128, 200)]),
            # filter numbers over all layers
            ("channel_size", [128]),
            # strides over all layers
            ("strides", [1])
            ])
        self.search = search

        self.first = ConvBNRelu(input_depth=1, output_depth=16, kernel=3, stride=2, # 输入通道
                        pad=1, no_bias=1, use_relu="relu", bn_type="bn")
        
        # self.rebult = MultiKernalConv(16, 16, 1000)
        self.sec = nn.Sequential(ConvBNRelu(input_depth=16, output_depth=128, kernel=3, stride=2, # 输入通道
                pad=1, no_bias=1, use_relu="relu", bn_type="bn"))
        self.avgpool = nn.AdaptiveAvgPool1d(200)

        self.searchOP = SearchOperation(self.SEARCH_SPACE, search,arc = arc, i = i)
    def forward(self, x, temperature):

        y = self.first(x)
        # y = self.rebult(y)
        y = self.sec(y)
        y = self.avgpool(y)
        y = self.searchOP(y, temperature)
        
        return y   

class Down(nn.Module):
    '''Downsampling module'''
    def __init__(self, stride=2,search = True, arc = None, i =0,index=1):
        super(Down, self).__init__()
        if index == 1:
            self.SEARCH_SPACE = OrderedDict([
                #### input shapes of all searched layers (considering with strides)
                ("input_shape", [( 256, 100)]),
                # filter numbers over all layers
                ("channel_size", [256]),
                # strides over all layers
                ("strides", [1])
                ])
        else:
            self.SEARCH_SPACE = OrderedDict([
                #### input shapes of all searched layers (considering with strides)
                ("input_shape", [( 512, 50)]),
                # filter numbers over all layers
                ("channel_size", [512]),
                # strides over all layers
                ("strides", [1])
                ])

        self.search = search
        self.searchOP = SearchOperation(self.SEARCH_SPACE, search, arc = arc, i = i)
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(self.SEARCH_SPACE["input_shape"][0][0]//2,self.SEARCH_SPACE["input_shape"][0][0], kernel_size=3, stride=1, padding=1)
            #ResBlock(in_channels, out_channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x, temperature):
        y = self.maxpool(x)
 
        y = self.searchOP(y, temperature)
   
        y = self.relu(y)
        
        return y
    
class Up(nn.Module):
    def __init__(self, bilinear=False, search = True,arc = [], i = 0,index = 3):
        super(Up, self).__init__()
        if index == 3:
            self.SEARCH_SPACE = OrderedDict([
                #### input shapes of all searched layers (considering with strides)
                ("input_shape", [( 256, 100)]),
                # filter numbers over all layers
                ("channel_size", [256]),
                # strides over all layers
                ("strides", [1])
                ])

        else:
            self.SEARCH_SPACE = OrderedDict([
                #### input shapes of all searched layers (considering with strides)
                ("input_shape", [( 128, 200)]),
                # filter numbers over all layers
                ("channel_size", [128]),
                # strides over all layers
                ("strides", [1])
                ])

        self.search = search
        self.searchOP = SearchOperation(self.SEARCH_SPACE, search, arc = arc, i = i)
       
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:
            self.up = nn.ConvTranspose1d(self.SEARCH_SPACE["input_shape"][0][0]*2, self.SEARCH_SPACE["input_shape"][0][0], kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv1d(self.SEARCH_SPACE["input_shape"][0][0]*2, self.SEARCH_SPACE["input_shape"][0][0], kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
    def forward(self, x1,x2,temperature):
        x1 = self.up(x1)

        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])    
        
        x = torch.cat([x2, x1], dim=1)
        y = self.conv(x)
        y = self.searchOP(y, temperature)
        y = self.relu(y)

        return y

class Mid_Model(nn.Module):
    def __init__(self, outChannel, bilinear=False, stride=1, search = True,arc = [], i = 0):
        super(Mid_Model, self).__init__()
        self.SEARCH_SPACE = OrderedDict([
            #### input shapes of all searched layers (considering with strides)
            ("input_shape", [( 128, 200)]),
            # filter numbers over all layers
            ("channel_size", [128]),
            # strides over all layers
            ("strides", [1])
            ])
        self.search = search
        self.searchOP = SearchOperation(self.SEARCH_SPACE, search, arc = arc, i = i)

        self.relu = nn.ReLU()
    def forward(self, x,temperature):

        y = self.searchOP(x, temperature)
        y = self.relu(y)

        return y


class SuperNet(nn.Module):
    def __init__(self, cnt_classes=22, operations_index=[0,1,2,3,4,5,6]):
        super(SuperNet, self).__init__()

        self.operations = nn.ModuleList()
        for index,idx in enumerate(operations_index):
            if idx == 0:
                self.operations.append(StartModule())
            elif idx == 1 or idx == 2:
                self.operations.append(Down(index=idx))
            elif idx == 3 or idx == 4:
                self.operations.append(Up(index=idx))
            elif idx == 5:
                self.operations.append(Mid_Model(128))
            elif idx == 6:
                self.operations.append(endModule(128, cnt_classes))
            else:
                raise ValueError("Invalid module index")

        self.sig = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool1d(100)
        self.last_stages = SuperNetOutWeights(cnt_classes)
        self.kan_module = nn.Sequential(OrderedDict([
            ("dropout", nn.Dropout(p=0.1)),
            ("flatten", Flatten()),
            ("linear", nn.Linear((cnt_classes*200), 22)),
            # ("kan", KAN(self.layers_hidden)),
            ("softmax", nn.Sigmoid())
        ]))

    def forward(self, x, temperature):
        x = x.unsqueeze(1)
        
        skips = [] # U-Net 结构：存储跳过连接的中间结果
        for i, op in enumerate(self.operations):
            if isinstance(op, Down): # 判断是否为down
                skips.append(x)
                x = op(x, temperature)
                  # 存储特征图以进行跳过连接
            elif isinstance(op, Up):
                skip_x = skips.pop(-1) # 从列表中移除的元素对象
                x = op(x, skip_x, temperature)  # 经过up来融合 x 和 skip_x
            else:
                x = op(x, temperature)

        logits_1 = self.kan_module(x)
        y = self.avg(x)
        logits, Weights = self.last_stages(x)
        logits = 0.9 * logits_1 + 0.1 * logits
        return logits, Weights

    def update_weights(self, feedback_scores):
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
        mse_loss = self.mse_loss(predictions, targets)
        # cross_loss = self.cross_entropy(predictions,targets)

        combined_loss = (self.alpha * mse_loss) + (self.beta * cosine_sim)
        return combined_loss
    
    def forward(self, predictions, targets, losses_ce, N):
        loss = self.weight_criterion(predictions, targets)
        losses_ce.update(loss.item(), N)

        return loss


""" 
The folowing is for retrain from scratch
"""

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# class FBNet(nn.Module):
#     def __init__(self, cnt_classes, arch_operations, operations_index, output_features=False):
#         super(FBNet, self).__init__()
        
#         self.output_features = output_features

#         # Avoid looping over operations and initialize only necessary modules
#         self.start_module = StartModule(search=False, arc=arch_operations, i=0) 
#         self.down1 = Down(search=False, arc=arch_operations, i=1, index=1)     
#         self.down2 = Down(search=False, arc=arch_operations, i=2, index=2)   
#         self.up1 = Up(search=False, arc=arch_operations, i=3, index=3)        
#         self.up2 = Up(search=False, arc=arch_operations, i=4, index=4)      
#         # self.mid_module = Mid_Model(outChannel=128, search=False)
#         self.end_module = endModule(128, cnt_classes*5, search=False, arc=arch_operations, i=5)

#         self.avg = nn.AdaptiveAvgPool1d(100)
#         self.last_stages = SuperNetOutWeights(cnt_classes*5)
#         self.sigmoid = nn.Sigmoid()
#         self.kan_module = nn.Sequential(OrderedDict([
#             ("dropout", nn.Dropout(p=0.1)),
#             ("flatten", Flatten()),
#             ("linear", nn.Linear(22000, 22)),
#             # ("kan", KAN(self.layers_hidden)),
#             ("softmax", nn.Sigmoid())
#         ]))

#     def forward(self, x, temperature=1.0):
#         x = x.unsqueeze(1)
    
#         skip1 = self.start_module(x, temperature)  
#         x = self.down1(skip1, temperature)  
#         skip2 = x                                  
#         x = self.down2(x, temperature)   
 
#         x = self.up1(x, skip2, temperature)       
#         x = self.up2(x, skip1, temperature)        
#         x = self.end_module(x, temperature)

#         logits_1 = self.kan_module(x)
#         y = self.avg(x)
#         logits, Weights = self.last_stages(x)
#         logits = 0.9 * logits_1 + 0.1 * logits
#         if self.output_features:
#             return logits, Weights.transpose(-1, -2)
#         else:
#             return logits


class FBNet(nn.Module):
    def __init__(self, cnt_classes, arch_operations, operations_index, output_features=False):
        super(FBNet, self).__init__()

        self.arch_operations = arch_operations
        self.output_features = output_features
        self.operations = nn.ModuleList()
        for index_id,idx in enumerate(operations_index):
            if idx == 0:
                self.operations.append(StartModule(search=False, arc=arch_operations,i=index_id))
            elif idx == 1 or idx == 2:
                self.operations.append(Down(search=False, arc=arch_operations, i=index_id, index=idx))
            elif idx == 3 or idx == 4:
                self.operations.append(Up(search=False, arc=arch_operations, i=index_id,index=idx))
            elif idx == 5:
                self.operations.append(Mid_Model(outChannel=128, search=False, arc=arch_operations, i=index_id))
            elif idx == 6:
                self.operations.append(endModule(128, cnt_classes, search=False, arc=arch_operations, i=index_id))
            else:
                raise ValueError("Invalid module index")

        self.sig = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool1d(100)
        self.last_stages = SuperNetOutWeights(cnt_classes)
        self.kan_module = nn.Sequential(OrderedDict([
            ("dropout", nn.Dropout(p=0.1)),
            ("flatten", Flatten()),
            ("linear", nn.Linear((cnt_classes*200), 22)),
            # ("kan", KAN(self.layers_hidden)),
            ("softmax", nn.Sigmoid())
        ]))

    def forward(self, x, temperature):
        x = x.unsqueeze(1)
        
        skips = [] # U-Net 
        for i, op in enumerate(self.operations):
            if isinstance(op, Down): # Determine whether it is down
                skips.append(x)
                x = op(x, temperature)
                  # Storing feature maps for skip connections
            elif isinstance(op, Up):
                skip_x = skips.pop(-1) # The element object removed from the list
                x = op(x, skip_x, temperature)  # Fusion of x and skip_x via up
            else:
                x = op(x, temperature)

        logits_1 = self.kan_module(x)
        y = self.avg(x)
        logits, Weights = self.last_stages(x)
        logits = 0.9 * logits_1 + 0.1 * logits
        if self.output_features:
            return logits, Weights.transpose(-1, -2)
        else:
            return logits

# test
# arch_operations =  ['trans', 'skip', 'ir_k5_e1', 'skip', 'graph', 'ir_k7_e1']  

# img = torch.randn(4,1000)
# model = SuperNet(cnt_classes=22, operations_index=operations_index)
# preds,_ = model(img, 0.1)
# print(preds.shape)

# img = torch.randn(4,1000)
# model = FBNet(cnt_classes=22,arch_operations=arch_operations, operations_index=operations_index,output_features=False)
# preds = model(img, 0.1)
# print(preds.shape)
