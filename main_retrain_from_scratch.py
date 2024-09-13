

from sklearn.preprocessing import LabelBinarizer
import torch
from torch import nn
import logging

from scipy.special import softmax
import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
import pandas as pd
from configs import CONFIG_FBNET, CONFIG_SUPERNET
from trainer_fbnet import TrainerFBNet
from utils import get_logger, load, weights_init, check_tensor_in_list
from trainer_supernet import TrainerSupernet
import torch.optim as optim
from trainer_supernet import TrainerSupernet
import shutil
import logging
from torch.utils.data import Dataset,DataLoader,Subset
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

'''
Model selection
'''
# from models_unet_1 import *
from models12 import *

'''
超网络后进行精调模型
'''

# 初始化日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def sample_architecture_from_the_supernet(save_path, hardsampling=True):
    """
    参数:
    hardsampling=True 表示获取权重最大的操作
                 =False 表示对权重应用 softmax 并从分布中采样
    """
    logger = get_logger('architecture_sampling.log')
    model = SuperNet(cnt_classes=22) # , operations_index=operations_index
    if torch.cuda.is_available():
        model.cuda()
    load(model, save_path)
    
    cnt_ops = len(CANDIDATE_BLOCKS)
    arch_operations = []

    def get_mixed_operations(module):
        mixed_operations = []
        for name, sub_module in module.named_modules():
            if isinstance(sub_module, MixedOperation):
                mixed_operations.append(sub_module)
        return mixed_operations

    mixed_operations = get_mixed_operations(model)
    
    for operation in mixed_operations:
        if hardsampling:
            arch_operations.append(CANDIDATE_BLOCKS[np.argmax(operation.thetas.detach().cpu().numpy())])
        else:
            rng = np.linspace(0, cnt_ops - 1, cnt_ops, dtype=int)
            distribution = softmax(operation.thetas.detach().cpu().numpy())
            arch_operations.append(CANDIDATE_BLOCKS[np.random.choice(rng, p=distribution)])
    
    logger.info("Sampled Architecture: " + " - ".join(arch_operations))
    return arch_operations

def prepare_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"创建了数据文件夹：{path}")
    else:
        logger.info(f"数据文件夹已经存在：{path}")

def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)
def load_and_normalize_data(data_path):
    data = np.load(data_path)
    normalized_data = min_max_normalize(data)  # 或者使用 z_score_normalize(data)
    return normalized_data



def test_and_evaluate(test_loader, model, device, save_path):
    if not save_path.endswith('/'):
        save_path += '/'
    
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs,_ = model(X,temperature=0.1)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y.argmax(dim=1)).sum().item()

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'测试集上的网络准确率: {accuracy:.2f}%')

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    output_save = pd.DataFrame(all_preds)
    outlabel_save = pd.DataFrame(all_labels)
    output_save.to_csv(save_path + 'model_output_data.csv', index=False, header=False)
    outlabel_save.to_csv(save_path + 'model_label_data.csv', index=False, header=False)

    pred_labels = all_preds.argmax(axis=1)
    true_labels = (all_labels> 0.5).astype(int)
    # Calculating confusion matrix
    pred_output_data = np.zeros_like(all_preds)
    for i in range(pred_labels.shape[0]):
        pred_output_data[i,pred_labels[i]] = 1 # 最大值的地方为1


    cm = confusion_matrix(true_labels.argmax(axis=1), pred_output_data.argmax(axis=1))
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(save_path + 'confusion_matrix.csv', index=True)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 10})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path + 'confusion_matrix.png')
    plt.clf()

    # Calculating ROC AUC for each class
    fpr_list = {}
    tpr_list = {}
    roc_auc_list = {}
    for i in range(22):
        fpr_list[i], tpr_list[i], _ = roc_curve(true_labels[:, i], pred_output_data[:, i])
        roc_auc_list[i] = auc(fpr_list[i], tpr_list[i])
        plt.plot(fpr_list[i], tpr_list[i], label=f'Class {i+1} (AUC = {roc_auc_list[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve per class')
    plt.legend(loc='lower right')
    plt.savefig(save_path + 'ROC_Curve.png')
    plt.clf()

    print('All evaluations are saved.')
    return accuracy

def fbnet_train(train_loader, valid_loader, test_loader, device,save_path,fbnet_save_path=None):
    manual_seed = 42 
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True
    arch_operations = sample_architecture_from_the_supernet(save_path, hardsampling=False)
    # arch_operations = ['GSAU', 'skip', 'ir_k7_e1', 'ir_k5_e1', 'graph', 'graph']

    logger = get_logger(CONFIG_FBNET['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_FBNET['logging']['path_to_tensorboard_logs'])

    # cnt_classes=22,operations_index=operations_index
    model = FBNet(cnt_classes=22,arch_operations=arch_operations,output_features=True).cuda()
    model = model.apply(weights_init)
    

    #### Loss and Optimizer
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=0.1,
    #                             momentum=CONFIG_FBNET['optimizer']['momentum'],
    #                             weight_decay=CONFIG_FBNET['optimizer']['weight_decay'])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-5) # model.parameters(), lr=0.002
    criterion = SupernetLoss().cuda()

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=CONFIG_FBNET['train_settings']['cnt_epochs'],
                                                        eta_min=0.00001, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                         milestones=[5,10,15],
    #                                         gamma=0.00001)  

    # pre_save_path = os.path.join(cycle_path, 'best_model_premodel.pth')
    #### Training Loop
    trainer = TrainerFBNet(criterion, optimizer, scheduler, logger, writer,fbnet_save_path)
    trainer.train_loop(train_loader, valid_loader, model)
    model.load_state_dict(torch.load(fbnet_save_path))


    accruracy = test_and_evaluate( test_loader,model,device,cycle_path)
    #把 accuracy 和 arch_operations 保存到文件txt
    
    with open(fbnet_save_path[:-4]+'.txt', 'w') as f:
        f.write(str(accruracy)+'\n')
        f.write(str(arch_operations)+'\n')
    
class CustomDataset(Dataset):
    def __init__(self, data_path, target_path):
        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]




if __name__ == "__main__":
    # features_train_path = "/home/fding/ADMIN/train_data.npy"
    # features_train_labels_path = "/home/fding/ADMIN/train_target.npy"
    # features_valid_path = "/home/fding/ADMIN/val_data.npy"
    # features_valid_labels_path = "/home/fding/ADMIN/val_target.npy"
    # features_test_path = "/home/fding/ADMIN/test_data.npy"
    # features_test_labels_path = "/home/fding/ADMIN/test_target.npy"
    
    #从训练集分20%当做验证集
   
    features_train_path = "/home/fding/ADMIN/train_data.npy"
    features_train_labels_path = "/home/fding/ADMIN/train_target.npy"
    features_valid_path = "/home/fding/ADMIN/val_data.npy"
    features_valid_labels_path = "/home/fding/ADMIN/val_target.npy"
    features_test_path = "/home/fding/ADMIN/test_data.npy"
    features_test_labels_path = "/home/fding/ADMIN/test_target.npy"

    # features_train = np.load(features_train_path)
    # features_train_labels = np.load(features_train_labels_path)
    # features_valid = np.load(features_valid_path)
    # features_valid_labels = np.load(features_valid_labels_path)

    # features_test = np.load(features_test_path)
    # features_test_labels = np.load(features_test_labels_path)
    # train_dataset = CustomDataset(features_train_path,features_train_labels_path)

    # valid_dataset = CustomDataset(features_valid_path,features_valid_labels_path)
    # test_dataset = CustomDataset(features_test_path,features_test_labels_path)

    # batchsize = 64
    # train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True)

    # valid_loader = DataLoader(valid_dataset,batch_size=batchsize,shuffle=False)
    
    # test_loader = DataLoader(test_dataset,batch_size=256,shuffle=False)

    # train_dataset = CustomDataset(features_train_path,features_train_labels_path)
    # total_samples = len(train_dataset)
    # train_size = int(0.8 * total_samples)
    # valid_size = total_samples - train_size

    # # 划分训练集和验证集索引
    # indices = list(range(total_samples))
    # train_indices = indices[:train_size]
    # valid_indices = indices[train_size:]
    # train_dataset = Subset(train_dataset, train_indices)
    # valid_dataset = Subset(train_dataset, valid_indices)

    # test_dataset = CustomDataset(features_test_path,features_test_labels_path)

    # batchsize = 64
    # train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True)

    # valid_loader = DataLoader(valid_dataset,batch_size=batchsize,shuffle=False)
    
    # test_loader = DataLoader(test_dataset,batch_size=256,shuffle=False)
    train_dataset = CustomDataset(features_train_path,features_train_labels_path)
    valid_dataset = CustomDataset(features_valid_path,features_valid_labels_path)
    test_dataset = CustomDataset(features_test_path,features_test_labels_path)

    batchsize = 256
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True,num_workers=8)

    valid_loader = DataLoader(valid_dataset,batch_size=batchsize,shuffle=False,num_workers=8)
    
    test_loader = DataLoader(test_dataset,batch_size=256,shuffle=False,num_workers=8)
      # 检查输出文件夹是否存在，如果存在则跳过，否则创建
    output_folder_path =r'output_12_1'
    prepare_output_folder(output_folder_path)
    for j in range(5):
    # j = 0
        cycle_path = os.path.join(output_folder_path, f'cycle--{j+1}')


        save_path = os.path.join(cycle_path, 'best_model_Supernet.pth')

        print('Cycle--------'+str(j+1))


        # 打印最终模型
        # print('Cycle--------'+str(j+1)+'--------Best Model')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = SuperNet(cnt_classes=22,output_features=False,SEARCH_SPACE=SEARCH_SPACE_8).to(device)

        # model = model.to(device)
        # state_dict = torch.load(save_path)
        # model.load_state_dict(state_dict)

        fbnet_save_path = os.path.join(cycle_path, 'best_model_FBNet.pth')
        fbnet_train(train_loader, valid_loader, test_loader, device,save_path,fbnet_save_path)

    
    # 保存真实标签与预测标签，制作混淆矩阵
    # 保存ROC曲线
    # 保存混淆矩阵
    # 保存最终测试集的准确率


    # # 合并数据
    # features_combined = np.concatenate((features_train, features_valid), axis=0)
    # labels_combined = np.concatenate((features_train_labels, features_valid_labels), axis=0)
    # # k-Fold Cross-Validation setup
    # k = 5  # Number of folds
    # kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # # Prepare the output folder
    # prepare_output_folder(output_folder_path)

    # # Model device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # #创建五个fold文件夹



    # fold = 0
    # for train_index, valid_index in kf.split(features_combined):
    #     fold += 1
    #     print(f"Cycle--------{fold}--------Training and Validation")
        
    #     # Splitting data
    #     features_train, features_valid = features_combined[train_index], features_combined[valid_index]
    #     labels_train, labels_valid = labels_combined[train_index], labels_combined[valid_index]
        
    #     # Create data loaders
    #     train_loader = get_dataloader(features_train, labels_train, batch_size=128, shuffle=True)
    #     valid_loader = get_dataloader(features_valid, labels_valid, batch_size=128, shuffle=False)
        
    #     # Model setup and training
    #     model = SuperNet(cnt_classes=22, output_features=False).to(device)

        
    #     j=2

    #     cycle_path = os.path.join(output_folder_path, f'cycle--{j+1}')

    #     os.makedirs(cycle_path+'/'+str(fold+1), exist_ok=True)
    #     save_path = os.path.join(cycle_path, 'best_model_Supernet.pth')
    #     print('Cycle--------'+str(j+1))


    #     # 打印最终模型
    #     print('Cycle--------'+str(j+1)+'--------Best Model')
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     model = SuperNet(cnt_classes=22,output_features=False).to(device)

    #     model = model.to(device)
    #     state_dict = torch.load('/home/fding/ADMIN/output/cycle--'+str(j+1)+'/best_model_Supernet.pth')
    #     model.load_state_dict(state_dict)
        
    #     fbnet_save_path = '/home/fding/ADMIN/output/cycle--'+str(j+1)+'/'+str(fold+1)+'/best_model_FBNet.pth'
    #     fbnet_train(train_loader, valid_loader, test_loader, device,save_path,fbnet_save_path)

    #     # Optionally, save best model for each fold
    #     print(f"fold--------{fold}--------Best Model Saved")
    #     # test( test_loader,model, device)
