

import os 
import time 
from datetime import datetime, timedelta

from model.segmentor_version2 import EncoderDecoder as SalsaNextAdapter
from losses.loss import FocalLosswithLovaszRegularizer
from data_loader.semantic_kitti import SemanticKITTI
from utils.pytorchtools import EarlyStopping
from train_adapter import Training 


import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter

def main():

    # args --------------------------------------------------------------------------------------
    name = ""

    # # gpu setting -----------------------------------------------------------------------------
    torch.cuda.manual_seed_all(777)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpu = list(range(torch.cuda.device_count()))
    num_workers = len(gpus.split(",")) * 2
    timeout=timedelta(seconds=864000)
    dist.init_process_group(backend='nccl', rank=0, world_size=1, timeout=timeout)

    # setting model params --------------------------------------------------------------------
    epochs = 2000
    batch_size = len(num_gpu)*10
    nclasses = 20 
    img_size = (256, 1024)

    # setting model ---------------------------------------------------------------------------
    model = SalsaNextAdapter(nclasses)
    model = DataParallel(model.to(device), device_ids=num_gpu)
    optimizer = Adam(model.to(device).parameters())
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    criterion = FocalLosswithLovaszRegularizer(ignore_idx=0)

    # setting data ----------------------------------------------------------------------------
    path = ''
    dataset = SemanticKITTI(data_path=path, shape=img_size, nclasses=nclasses, mode='train', front=True) # 360 
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size)

    # create dir for weight --------------------------------------------------------------------------------
    configs = "{}_batch{}_epoch{}_{}_{}".format(path.split('/')[4], batch_size, epochs, str(criterion).split('(')[0], str(optimizer).split( )[0])
    print("Configs:", configs)
    now = time.strftime('%m%d_%H%M') 
    model_path = os.path.join("weights", configs, name+str(now))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    earlystop = EarlyStopping(patience=10, verbose=True, path=os.path.join(model_path, 'earlystop.pt'))

    # write log --------------------------------------------------------------------------------
    metrics = {'t_loss':[], 'v_loss':[], 't_miou':[], 'v_miou':[]}

    if not os.path.exists(os.path.join(model_path, 'samples')):
        os.makedirs(os.path.join(model_path, 'samples'))
        os.makedirs(os.path.join(model_path, 'samples', 'out1'))
        os.makedirs(os.path.join(model_path, 'samples', 'out2'))
        os.makedirs(os.path.join(model_path, 'samples', 'out2', 'train'))
        os.makedirs(os.path.join(model_path, 'samples', 'out2', 'val'))
        
    if not os.path.exists(os.path.join(model_path, 'train')):
        os.makedirs(os.path.join(model_path, 'train'))
    if not os.path.exists(os.path.join(model_path, 'val')):
        os.makedirs(os.path.join(model_path, 'val'))
 
    writer_train = SummaryWriter(log_dir=os.path.join(model_path, 'train'))
    writer_val = SummaryWriter(log_dir = os.path.join(model_path, 'val'))

    with open(f'{model_path}/result.csv', 'a') as epoch_log:
        epoch_log.write('\nepoch\ttrain loss\tval loss\ttrain mIoU\tval mIoU')

    t_s = datetime.now()
    print(f'\ntrain start time : {t_s}')

    t = Training(model, epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_path, earlystop, device, metrics, writer_train, writer_val)
    t.train()

    print(f'\n[train time information]\n\ttrain start time\t{t_s}\n\tend of train\t\t{datetime.now()}\n\ttotal train time\t{datetime.now()-t_s}')


if __name__ == "__main__":
    main()
