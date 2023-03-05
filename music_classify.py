import os
import time
from datetime import datetime
import argparse
from tensorboardX import SummaryWriter
from torch import optim
import torch
from torch.utils.data import DataLoader
from model import Classifier
from dataloader import MusicDataset
import logging
import torch.nn as nn
# from logger import setup_logger
from torch.cuda.amp import autocast as autocast # for reducing the allocation of GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
logger = logging.getLogger()


def get_time():
    now_time = datetime.now()
    return str(now_time.year) + '-' + str(now_time.month) + '-' + str(now_time.day) + '-' + str(
        now_time.hour) + '-' + str(now_time.minute) + '-' + str(now_time.second)


def make_parses():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '--epoch',
        default=30,
        type=int
    )
    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int
    )
    parser.add_argument(
        '--batch-size',
        default=3,
        type=int
    )
    parser.add_argument(
        '--model-name',
        default='CP',
        type=str,
        help='Optional: CP, JC, JP'
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str
    )
    parser.add_argument(
        '--gamma',
        default=1,
        type=float
    )
    parser.add_argument(
        '--sigma',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--lamb',
        default=10,
        type=float
    )
    parser.add_argument(
        '--sample_size',
        default=50,
        type=int
    )
    parser.add_argument(
        '--save-frq',
        default=1000,
        type=int
    )
    parser.add_argument(
        '--log_frq',
        default=1,
        type=int
    )
    parser.add_argument(
        '--lr',
        default=0.00002,
        type=float
    )
    parser.add_argument(
        '--wd', '--weight-decay',
        default=0,
        type=float
    )
    parser.add_argument(
        '--data-mode',
        default='full',
        type=str,
        help='Optional: full, '
    )
    return parser.parse_args()


def train():
    # ------- set the directory of training dataset --------
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename='{}.log'.format(get_time()),
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format='%(asctime)s - %(message)s'  # 日志格式
                        )

    args = make_parses()
    model_name = args.model_name  # JC CP JP.
    writer = SummaryWriter(comment=model_name + str(time.time()))

    data_dir = os.path.join(os.getcwd(), 'data' + os.sep)
    print('路径：' + data_dir)
    now_time = datetime.now()
    now_mon = now_time.month
    now_day = now_time.day
    now_hour = now_time.hour
    now_minute = now_time.minute
    model_dir = os.path.join(os.getcwd(), 'saved_classifier', str(now_mon) + '-' + str(now_day) + '-'
                             + str(now_hour) + '-' + str(now_minute) + os.sep)
    print(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    epoch_num = args.epoch
    batch_size_train = args.batch_size
    # gamma = args.gamma
    save_frq = args.save_frq
    log_frq = args.log_frq

    logger.info("Hyperparameter: lr:{} wd:{} batch_size:{}".format(args.lr, args.wd, args.batch_size))

    music_dataset = MusicDataset(data_dir, train_mode=model_name, data_mode=args.data_mode, is_train='train')
    print(len(music_dataset))
    train_num = len(music_dataset)
    # music_dataset._get_name(1)

    logger.info("Train data contains 2*{} items".format(train_num))
    music_dataloader = DataLoader(
        music_dataset, batch_size=batch_size_train, shuffle=False, num_workers=0)
    # setup_logger(model_dir)

    logger.info("Load model with mode {}".format(model_name))
    logger.info("Model Args: sigma:{} sample_size:{} lamb:{}".format(args.sigma, args.sample_size, args.lamb))
    # ------- define model --------
    model = Classifier(dim=64)
    # model = CycleGAN(sigma=args.sigma, sample_size=args.sample_size, lamb=args.lamb, mode='train')
    if args.resume:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if 'model_name' in checkpoint.keys():
            assert model_name == checkpoint['model_name']
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    logger.info("---Define Optimizer---")
    lr = args.lr
    wd = args.wd
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=wd)

    # ------- training process --------
    logger.info("---Start Training---")
    ite = 0
    loss = 0.0
    ite_num4val = 0
    torch.autograd.set_detect_anomaly(True)
    start = time.time()
    for epoch in range(args.start_epoch, epoch_num):
     # with autocast():
        model.train()
        for i, data in enumerate(music_dataloader):
            ite = ite + 1
            ite_num4val = ite_num4val + 1
            real_a = data['bar_a']
            real_a = torch.FloatTensor(real_a)
            real_a = real_a.to(device)
            y_p = model(real_a)
            y_t = real_a[0] #应该改为对应的真实标签
            y_t = torch.FloatTensor([[0,1],[1,0],[1,0]]).to(device)
            loss_fn = nn.BCELoss()
            loss = loss_fn(y_p, y_t)

            # real_a, real_b, real_mixed = data['bar_a'], data['bar_b'], data['bar_mixed']
            # real_a = torch.FloatTensor(real_a)
            # real_b = torch.FloatTensor(real_b)
            # real_mixed = torch.FloatTensor(real_mixed)
            #
            # real_a = real_a.to(device)
            # real_b = real_b.to(device)
            # real_mixed = real_mixed.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()


            # cycle_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss, \
            # d_A_all_loss, d_B_all_loss = model(real_a, real_b, real_mixed)
            # # Generator loss
            # g_loss = g_A2B_loss + g_B2A_loss - cycle_loss
            #
            # # Discriminator loss
            # d_loss = d_A_loss + d_B_loss
            #
            # d_all_loss = d_A_all_loss + d_B_all_loss
            # D_loss = d_loss + gamma * d_all_loss
            # g_A2B_loss.backward(retain_graph=True)
            # g_B2A_loss.backward(retain_graph=True)
            #
            # d_A_loss.backward(retain_graph=True)
            # d_B_loss.backward(retain_graph=True)
            #
            # d_A_all_loss.backward(retain_graph=True)
            # d_B_all_loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            loss += loss.data.item()
            # g_running_loss += g_loss.data.item()
            # d_running_loss += D_loss.data.item()

            # writer.add_scalar('cycle_loss', cycle_loss, global_step=ite)
            # writer.add_scalar('g_A2B_loss', g_A2B_loss, global_step=ite)
            # writer.add_scalar('g_B2A_loss', g_B2A_loss, global_step=ite)
            # writer.add_scalar('d_A_loss', d_A_loss, global_step=ite)
            # writer.add_scalar('d_B_loss', d_B_loss, global_step=ite)
            # writer.add_scalar('d_A_all_loss', g_loss, global_step=ite)
            # writer.add_scalar('d_B_all_loss', g_loss, global_step=ite)
            # writer.add_scalar('d_all_loss', g_loss, global_step=ite)
            # writer.add_scalar('D_loss', D_loss, global_step=ite)

            # del g_A2B_loss, g_B2A_loss, g_loss, d_A_loss, d_B_loss, d_loss, \
            #     d_A_all_loss, d_B_all_loss, d_all_loss, D_loss
            if i % log_frq == 0:
                end = time.time()
                logger.info(
                    "[epoch: %3d/%3d, batch: %5d/%5d, ite: %d, time: %3f, loss: %3f]"
                    % (epoch+1, epoch_num, i*batch_size_train, train_num, ite, end-start, loss/ite_num4val)
                )
                # logger.info("[epoch: %3d/%3d, "
                #             "batch: %5d/%5d, "
                #             "ite: %d, "
                #             "time: %3f] "
                #             "g_loss : %3f, "
                #             "d_loss : %3f " % (
                #                 epoch + 1, epoch_num,
                #                 (i) * batch_size_train, train_num,
                #                 ite,
                #                 end - start,
                #                 g_running_loss / ite_num4val,
                #                 d_running_loss / ite_num4val))
                start = end

            if ite % save_frq == 0:
                saved_model_name = model_dir + model_name + "_itr_%d_LOSS_%3f.pth" % (
                    ite, loss / ite_num4val)
                torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model.state_dict()},
                    saved_model_name)
                logger.info("saved model {}".format(saved_model_name))
                loss = 0.0
                model.train()
                ite_num4val = 0
def test():
    pass

if __name__ == "__main__":
    train()
