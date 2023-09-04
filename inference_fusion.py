
import os, cv2, time, yaml
import numpy as np
from datetime import timedelta
from einops import rearrange

from model.segmentor_version2 import EncoderDecoder as SalsaNextAdapter
from model.segmentor_scunet import EncoderDecoder as SCUNet 
from data_loader.semantic_kitti import SemanticKITTI
from utils.load import read_model
from utils.logs import AverageMeter, ProgressMeter
from utils.metrics import IOUEval

import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader

def convert_color(arr, color_dict):
    result = np.zeros((*arr.shape, 3))
    for i in color_dict:
        j = np.where(arr==i)

        try:
            xs, ys = j[0], j[1]
        except:
            xs = j[0]

        if len(xs) == 0:
            continue
        for x, y in zip(xs, ys):
            result[x,y,2] = color_dict[i][0]
            result[x,y,1] = color_dict[i][1]
            result[x,y,0] = color_dict[i][2]

    return result

def inference(model1, model2, test_loader, nclasses, save_path, device):

    miou_run = AverageMeter('mIoU', ':.4f')
    car_run = AverageMeter('car', ':.4f')
    bicycle_run = AverageMeter('bicycle', ':.4f')
    motorcycle_run = AverageMeter('motorcycle', ':.4f')
    truck_run = AverageMeter('trunk', ':.4f')
    other_vehicle_run = AverageMeter('other_vehicle', ':.4f')
    person_run = AverageMeter('person', ':.4f')
    bicyclist_run = AverageMeter('bicyclist', ':.4f')
    motorcyclist_run = AverageMeter('motorcyclist', ':.4f')
    road_run = AverageMeter('road', ':.4f')
    parking_run = AverageMeter('parking', ':.4f')
    sidewalk_run = AverageMeter('sidewalk', ':.4f')
    other_ground_run = AverageMeter('other_ground', ':.4f')
    building_run = AverageMeter('building', ':.4f')
    fence_run = AverageMeter('fence', ':.4f')
    vegetation_run = AverageMeter('vegetation', ':.4f')
    trunk_run = AverageMeter('trunk', ':.4f')
    terrain_run = AverageMeter('terrain', ':.4f')
    pole_run = AverageMeter('pole', ':.4f')
    traffic_sign_run = AverageMeter('traiffic_sign', ':.4f')
    progress = ProgressMeter(len(test_loader), [miou_run])
    iou = IOUEval(nclasses, ignore=0)
    cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    color_dict = CFG['color_map']
    learning_map = CFG['learning_map']
    learning_map_inv = CFG['learning_map_inv']
    color_dict = {learning_map[key]:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

    model1.eval()
    model2.eval()
    
    total_label_check = torch.zeros([len(test_loader), 20])
    j = 0 
    for iter, batch in enumerate(test_loader):
        label_check = torch.zeros(20)
        inputs = rearrange(batch['rdm'].to(device), 'b1 b2 c h w -> (b1 b2) c h w')
        labels = rearrange(batch['3d_label'].to(device), 'b1 b2 h w -> (b1 b2) h w') # rgb 3channel
        
        total_label_check[j] = label_check
        j += 1
        
        imgs = batch['img']
        bs = inputs.size(0)

        with torch.no_grad():
            out_rgb, _, _, _ = model1(inputs)
            out_segment = model2(out_rgb, inputs)
        
        iou.addBatch(torch.argmax(out_segment, 1), labels)
        miou, per_iou = iou.getIoU()
                                  # origin 0.3125
        miou_run.update(miou.item(), bs) # 0.3144
        
        ignore_add = []
        for i in range(nclasses):
            cnt = (labels==i).sum()
            if cnt == 0:
                ignore_add.append(i)

        if 1 not in ignore_add: 
            car_run.update(per_iou[1].item(), bs)
        if 2 not in ignore_add:
            bicycle_run.update(per_iou[2].item(), bs)
        if 3 not in ignore_add:
            motorcycle_run.update(per_iou[3].item(), bs)
        if 4 not in ignore_add:
            truck_run.update(per_iou[4].item(), bs)
        if 5 not in ignore_add:
            other_vehicle_run.update(per_iou[5].item(), bs)
        if 6 not in ignore_add:
            person_run.update(per_iou[6].item(), bs)
        if 7 not in ignore_add:
            bicyclist_run.update(per_iou[7].item(), bs)
        if 8 not in ignore_add:
            motorcyclist_run.update(per_iou[8].item(), bs)
        if 9 not in ignore_add:
            road_run.update(per_iou[9].item(), bs)
        if 10 not in ignore_add:
            parking_run.update(per_iou[10].item(), bs)
        if 11 not in ignore_add:
            sidewalk_run.update(per_iou[11].item(), bs)
        if 12 not in ignore_add:
            other_ground_run.update(per_iou[12].item(), bs)
        if 13 not in  ignore_add:
            building_run.update(per_iou[13].item(), bs)
        if 14 not in ignore_add:
            fence_run.update(per_iou[14].item(), bs)
        if 15 not in ignore_add:
            vegetation_run.update(per_iou[15].item(), bs)
        if 16 not in ignore_add:
            trunk_run.update(per_iou[16].item(), bs)
        if 17 not in ignore_add:
            terrain_run.update(per_iou[17].item(), bs)
        if 18 not in ignore_add:
            pole_run.update(per_iou[18].item(), bs)
        if 19 not in ignore_add:
            traffic_sign_run.update(per_iou[19].item(), bs)
        progress.display(iter)
        
        all_outrgb = torch.cat([out_rgb[0,:,:,:512], out_rgb[1], out_rgb[2,:,:,512:]], axis=-1)
        all_outseg = torch.cat([out_segment[0,:,:,:512], out_segment[1], out_segment[2,:,:,512:]], axis=-1)
        all_input = torch.cat([inputs[0,:,:,:512], inputs[1], inputs[2,:,:,512:]], axis=-1).cpu().detach().numpy()

        cv2.imwrite(f'{save_path}/vis_samples/input.png', rearrange(all_input, 'c h w -> h w c'))
        cv2.imwrite(f'{save_path}/vis_samples/rgb.png', rearrange(all_outrgb.cpu().detach().numpy()*255, 'c h w -> h w c'))
        cv2.imwrite(f'{save_path}/vis_samples/segment.png',convert_color(torch.argmax(all_outseg, 0).cpu().detach().numpy(), color_dict))

        cv2.imwrite(f'{save_path}/vis_samples/img.png', cv2.resize(rearrange(imgs[0].detach().numpy(), 'c h w -> h w c'), (1024, 256)))

        [cv2.imwrite(f'{save_path}/vis_samples/input_split{str(i)}.png', rearrange(inputs[i]*10, 'c h w -> h w c').cpu().detach().numpy()) for i in range(inputs.shape[0])]
        [cv2.imwrite(f'{save_path}/vis_samples/rgb_split{str(i)}.png', rearrange(out_rgb[i]*255, 'c h w -> h w c').cpu().detach().numpy()) for i in range(out_rgb.shape[0])]
        [cv2.imwrite(f'{save_path}/vis_samples/segment_split{str(i)}.png', convert_color(torch.argmax(out_segment[i],0).cpu().detach().numpy(), color_dict)) for i in range(out_segment.shape[0])]
        
        # save per data
        # cv2.imwrite(f'{save_path}/vis_samples/rgb/{iter}.png', rearrange(out_rgb[-1]*255, 'c h w -> h w c').cpu().detach().numpy())
        # cv2.imwrite(f'{save_path}/vis_samples/segment/{iter}.png', torch.argmax(out_segment,1)[-1].cpu().detach().numpy()*10)
        # cv2.imwrite(f'{save_path}/vis_samples/segment/{iter}_img.png', rearrange(imgs[0], 'c h w -> h w c'))

    miou = miou_run.avg 
    car, bicycle, motorcycle, truck, other_vehicle, = car_run.avg, bicycle_run.avg, motorcycle_run.avg, truck_run.avg, other_vehicle_run.avg
    person, bicyclist, motorcyclist, road, parking = person_run.avg, bicyclist_run.avg, motorcyclist_run.avg, road_run.avg, parking_run.avg
    sidewalk, other_ground, building, fence, vegetation = sidewalk_run.avg, other_ground_run.avg, building_run.avg, fence_run.avg, vegetation_run.avg,
    trunk, terrain, pole, traffic_sign = trunk_run.avg, terrain_run.avg, pole_run.avg, traffic_sign_run.avg

    print(f'\nmIoU\t\t\t{miou} ---------------------------------')
    print('\ncar\t\t\t\t{:.4f}'.format(car))
    print('bicycle\t\t\t\t{:.4f}'.format(bicycle))
    print('motorcycle\t\t\t{:.4f}'.format(motorcycle))
    print('truck\t\t\t\t{:.4f}'.format(truck))
    print('other_vehicle\t\t\t{:.4f}'.format(other_vehicle))
    print('person\t\t\t\t{:.4f}'.format(person))
    print('bicyclist\t\t\t{:.4f}'.format(bicyclist))
    print('motorcyclelist\t\t\t{:.4f}'.format(motorcyclist))
    print('road\t\t\t\t{:.4f}'.format(road))
    print('parking\t\t\t\t{:.4f}'.format(parking))
    print('sidewalk\t\t\t{:.4f}'.format(sidewalk))
    print('other ground\t\t\t{:.4f}'.format(other_ground))
    print('building\t\t\t{:.4f}'.format(building))
    print('fence\t\t\t\t{:.4f}'.format(fence))
    print('vegetation\t\t\t{:.4f}'.format(vegetation))
    print('trunk\t\t\t\t{:.4f}'.format(trunk))
    print('terrain\t\t\t\t{:.4f}'.format(terrain))
    print('pole\t\t\t\t{:.4f}'.format(pole))
    print('traffic_sign\t\t\t{:.4f}'.format(traffic_sign))

    print('\nEND\n')      

    with open(f'{save_path}/result.txt','w') as f:
        f.write(f'\ntotal mIoU\t\t\t: {miou:.4f}\n\n')
        f.write('\nIoU per Class Result-----------------\n\n')
        f.write(f'car\t\t\t\t: {car:.4f}\n')
        f.write(f'bicycle\t\t\t: {bicycle:.4f}\n') 
        f.write(f'motorcycle\t\t: {motorcycle:.4f}\n') 
        f.write(f'truck\t\t\t: {truck:.4f}\n') 
        f.write(f'otehr vehicle\t: {other_vehicle:.4f}\n') 
        f.write(f'person\t\t\t: {person:.4f}\n') 
        f.write(f'bicyclist\t\t: {bicyclist:.4f}\n') 
        f.write(f'motorcyclelist\t: {motorcyclist:.4f}\n') 
        f.write(f'road\t\t\t: {road:.4f}\n') 
        f.write(f'parking\t\t\t: {parking:.4f}\n') 
        f.write(f'sidewalk\t\t: {sidewalk:.4f}\n') 
        f.write(f'other ground\t: {other_ground:.4f}\n') 
        f.write(f'building\t\t: {building:.4f}\n') 
        f.write(f'fence\t\t\t: {fence:.4f}\n') 
        f.write(f'vegetation\t\t: {vegetation:.4f}\n') 
        f.write(f'trunk\t\t\t: {trunk:.4f}\n') 
        f.write(f'terrain\t\t\t: {terrain:.4f}\n') 
        f.write(f'pole\t\t\t: {pole:.4f}\n') 
        f.write(f'traffic_sign\t: {traffic_sign:.4f}\n') 

def main():

    # gpu setting -----------------------------------------------------------------------------
    torch.cuda.manual_seed_all(777)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpu = list(range(torch.cuda.device_count()))
    num_workers = len(gpus.split(",")) * 2
    timeout = timedelta(seconds=864000)
    dist.init_process_group(backend='nccl', rank=0, world_size=1, timeout=timeout)
    
    batch_size = len(num_gpu)*1
    nclasses = 20
    img_shape = (256, 1024*2)
    name = ''

    # setting model1 
    model1_path = '/vit-adapter-kitti/jyjeon/weights/_SC_UNet/kitti_batch60_epoch2000_MSELoss_Adam/output255_feature6_level123_head2_37_RDM_Dresblock_10sobel2_0809_1429/earlystop.pt'
    model1 = SCUNet(dim=48, mode='feature6', mode2='level123', resblock=True)
    model1 = DataParallel(model1.to(device), device_ids=num_gpu)
    model1, _  = read_model(model1, model1_path, True)

    # setting model2
    model2_path = '/vit-adapter-kitti/jyjeon/weights/kitti_batch24_epoch2000_FocalLosswithLovaszRegularizer_Adam/fusion_vesion3_0901_1121/earlystop.pt'
    model2 = SalsaNextAdapter(nclasses)
    model2 = DataParallel(model2.to(device), device_ids=num_gpu)
    model2, _ = read_model(model2, model2_path, True)

    # setting data loader
    dataset_path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences'
    dataset = SemanticKITTI(data_path=dataset_path, shape=img_shape, nclasses=nclasses, mode='test', front=False, split=True)
    test_loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    # setting save result dir
    save_path = os.path.join(('/').join(model2_path.split('/')[:-1]), f"{name}_{time.strftime('%m%d_%H%M')}")

    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, 'vis_samples'))
        os.makedirs(os.path.join(save_path, 'vis_samples', 'rgb'))
        os.makedirs(os.path.join(save_path, 'vis_samples', 'segment'))
    
    inference(model1, model2, test_loader, nclasses, save_path, device)
    


if __name__ == "__main__":
    main()
