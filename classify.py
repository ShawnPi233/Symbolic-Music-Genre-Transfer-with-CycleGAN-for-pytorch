import os
import torch
import argparse
from torch.utils.data import DataLoader
from dataloader import MusicDataset, CustomDataset
from utils import save_midis, to_binary
from model import CycleGAN
from datetime import datetime
import logging
logger = logging.getLogger()
# from torch.cuda.amp import autocast as autocast # for reducing the allocation of GPU
def get_time():
    now_time = datetime.now()
    return str(now_time.year) + '-' + str(now_time.month) + '-' + str(now_time.day) + '-' + str(
        now_time.hour) + '-' + str(now_time.minute) + '-' + str(now_time.second)
def make_parses():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '--data-dir',
        default=None,
        type=str
    )
    parser.add_argument(
        '--model-dir',
        default = r'saved_models/JC/JC_itr_32000_G_1.386985_D_0.461608.pth',
        # default = r'saved_models/CP_itr_216000_G_1.367802_D_0.440168.pth',
        # default = r'saved_models/JC_itr_107000_G_1.382960_D_0.457389.pth',
        # default = r'saved_models/CP/CP_itr_197000_G_1.354145_D_0.452297.pth',
        # default = r'saved_models/CP/CP_itr_158000_G_1.365788_D_0.455798.pth',
        # default = r'saved_models/CP/CP_itr_126000_G_1.365276_D_0.461854.pth',
        # default=r'saved_models/CP/CP_itr_47000_G_1.370169_D_0.470722.pth',
        # default=r'saved_models\CP-1-8-1-2\CP_itr_64000_G_1.347712_D_0.441307.pth',
        type=str
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int
    )
    parser.add_argument(
        '--model-name',
        default='JC',
        type=str,
        help='Optional: CP, JC, JP'
    )
    parser.add_argument(
        '--test-mode',
        default='A2B',
        type=str
    )
    return parser.parse_args()

def test():
    # JC CP JP.
    args = make_parses()
    model_name = args.model_name
    mode = args.test_mode
    model_dir = args.model_dir
    data_dir = args.data_dir if args.data_dir else os.path.join(os.getcwd(), 'data' + os.sep)
    now_time = datetime.now()
    now_mon = now_time.month
    now_day = now_time.day
    now_hour = now_time.hour
    now_minute = now_time.minute
    now_second = now_time.second
    save_dir = os.path.join(os.getcwd(), 'test'+ '-' + model_name +
    '-'+ str(now_mon)+ '-' +str(now_day)+ '-' +str(now_hour)+ '-' +str(now_minute)+ '-' + str(now_second)+
    os.sep)
    # save_dir = os.path.join(os.getcwd(), 'test' + os.sep) #风格迁移后音乐保存路径

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename=save_dir + '/{}.log'.format('test_' + model_name+'_' + get_time()),
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format='%(asctime)s - %(message)s'  # 日志格式
                        )
    if args.data_dir is None:
        music_dataset = MusicDataset(data_dir, train_mode='CP', data_mode='full', is_train='test')
    else:
        music_dataset = CustomDataset(data_dir)

    music_dataloader = DataLoader(
        music_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("test dataset contains {} items".format(len(music_dataset)))
    # ------- 3. define model --------
    net = CycleGAN(mode=mode)
    checkpoint = torch.load(args.model_dir)
    if 'model_name' in checkpoint.keys():
        assert checkpoint['model_name'] == model_name
    if 'state_dict' in checkpoint.keys():
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    # ------- 5. training process --------
    print("---start testing...")
    logger.info('load model:{}'.format(model_dir))
    for i, data in enumerate(music_dataloader):
      # with autocast():
        real_a, real_b, real_mixed = data['bar_a'], data['bar_b'], data['bar_mixed']
        real_a = torch.FloatTensor(real_a)
        real_b = torch.FloatTensor(real_b)
        real_mixed = torch.FloatTensor(real_mixed)

        if torch.cuda.is_available():
            real_a = real_a.cuda()
            real_b = real_b.cuda()
            real_mixed = real_mixed.cuda()

        transfered, cycle = net(real_a, real_b, real_mixed)
        transfered = transfered.permute(0, 2, 3, 1) # torch.permute函数用于张量维度换位，即依次将第0，2，3，1维的张量替换当前维度（依次为 0，1，2，3）张量
        cycle = cycle.permute(0, 2, 3, 1)

        trans_np = to_binary(transfered.detach().cpu().numpy())
        cycle_np = to_binary(cycle.detach().cpu().numpy())

        name = music_dataset._get_name(data['baridx'])
        print(type(name))
        print('save to '+ save_dir + name + '_transfered.mid')
        save_midis(trans_np, save_dir + name + '_transfered.mid')
        save_midis(real_a.permute(0, 2, 3, 1).detach().cpu().numpy(),
                   save_dir + name + '_origin.mid')
        save_midis(cycle_np, save_dir + name + '_cycle.mid')

if __name__ == '__main__':
    test()
