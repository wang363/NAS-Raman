import torch
from torch import nn
import logging
from scipy.special import softmax
import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from trainer_fbnet import TrainerFBNet
from utils import get_logger, load, weights_init, check_tensor_in_list
from configs import CONFIG_SUPERNET
from trainer_supernet import TrainerSupernet
import torch.optim as optim
from trainer_supernet import TrainerSupernet
import shutil
import logging
from torch.utils.data import Dataset,DataLoader,Subset
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split

'''
Model selection
'''
# from models_unet_1 import CANDIDATE_BLOCKS,  SuperNet, SupernetLoss, operations_index
from models12 import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
Training of the front network and the super network
The front network obtains the feature data set and merges it with the training set for feature fusion and then conducts training
'''


def nor_max(arrlist):
    max_val = np.max(np.abs(arrlist) )
    #arrlist[arrlist<0] = 0
    arr_norm = arrlist / max_val
    return arr_norm
def add_noise(data, noise_level=0.02):
    noise = torch.randn_like(data) * noise_level
    return data + noise

import torch
from torch.utils.tensorboard import SummaryWriter

def train_supernet(train_w_loader,train_thetas_loader,valid_loader,config, device,save_path):
    logger = get_logger(config['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=config['logging']['path_to_tensorboard_logs'])

    # Initialize the model
    model = SuperNet(cnt_classes=22).to(device) # , operations_index=operations_index
    model.apply(weights_init)

    criterion = SupernetLoss().to(device)
    
    thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]



    # w_optimizer = torch.optim.SGD(params=params_except_thetas, lr=0.01, momentum=0.8, weight_decay=1e-4)
    # w_optimizer = torch.optim.SGD(params=params_except_thetas, lr=0.1, momentum=config['optimizer']['w_momentum'], weight_decay=config['optimizer']['w_weight_decay'])
    w_optimizer = torch.optim.AdamW(params=params_except_thetas, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    theta_optimizer = torch.optim.AdamW(thetas_params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    # theta_optimizer = torch.optim.SGD(params=params_except_thetas, lr=0.1, momentum=0.9, weight_decay=1e-4)
     
    # Learning Rate Scheduler
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, T_max=config['train_settings']['cnt_epochs'], last_epoch=-1)
    # w_scheduler = torch.optim.lr_scheduler.StepLR(w_optimizer, step_size=5, gamma=0.1)                                                                                                                                      
  
    # Training Logic
    trainer = TrainerSupernet( criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer,save_path)
    trainer.train_loop(train_w_loader, train_thetas_loader, valid_loader, model)


def prepare_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"创建了数据文件夹：{path}")
    else:
        logger.info(f"数据文件夹已经存在：{path}")
        
def test(test_loader, model,device):

    temperature = CONFIG_SUPERNET['train_settings']['init_temperature']
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs,_ = model(X,temperature)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y.argmax(dim=1)).sum().item()

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'测试集上的网络准确率: {accuracy:.2f}%')

def setup_seed_and_device(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data_path, target_path):
        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label
def main():
    device = setup_seed_and_device()
    features_train_path = "/home/fding/ADMIN/train_data.npy"
    features_train_labels_path = "/home/fding/ADMIN/train_target.npy"
    features_valid_path = "/home/fding/ADMIN/val_data.npy"
    features_valid_labels_path = "/home/fding/ADMIN/val_target.npy"
    features_test_path = "/home/fding/ADMIN/test_data.npy"
    features_test_labels_path = "/home/fding/ADMIN/test_target.npy"

    train_dataset = CustomDataset(features_train_path,features_train_labels_path)
    valid_dataset = CustomDataset(features_valid_path,features_valid_labels_path) 
    test_dataset = CustomDataset(features_test_path,features_test_labels_path)

    batchsize = 256
    train_w_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True, num_workers=8)
    train_thetas_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset,batch_size=batchsize,shuffle=False, num_workers=8)
    
    test_loader = DataLoader(test_dataset,batch_size=128,shuffle=False)
    # Check if the output folder exists, if so skip it, otherwise create it
    output_folder_path =r'./output_12_1'
    prepare_output_folder(output_folder_path)

    for j in range(5):
        cycle_path = os.path.join(output_folder_path, f'cycle--{j+1}')

        if not os.path.exists(cycle_path):
            os.makedirs(cycle_path)
        save_path = os.path.join(cycle_path, 'best_model_Supernet.pth')

        print('Cycle--------'+str(j+1))
        # Train SuperNet
        train_supernet(train_w_loader,train_thetas_loader,valid_loader,CONFIG_SUPERNET, device,save_path)
        # train_supernet(weighted_train_loader,weighted_train_loader,weighted_valid_loader,CONFIG_SUPERNET, device,save_path)
        print('Cycle--------'+str(j+1)+'--------Done')
        print('Cycle--------'+str(j+1)+'--------Best Model')
        model = SuperNet(cnt_classes=22).to(device) # , operations_index=operations_index
        model.load_state_dict(torch.load(save_path))
        test(test_loader, model,device)

if __name__ == "__main__":
    main()
