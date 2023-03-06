# import torch
# beta = torch.linspace(0.001,0.2,100)
# alpha = 1. - beta
# alpha_bar = torch.cumprod(alpha,dim=0).to('cuda')
# def f()->int:
#     global alpha_bar
#     return alpha_bar
# b = f()
# print(b)
import os
import glob
def glob_demo():
    dir = r'D:\codes\papers_code\Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch-main'
    import glob
    a = glob.glob(dir + '\*' + '.py')
    b = glob.glob(dir + '\*' + '.log')
    c = a+b
    print(c)

def del_func():
    a = 1  # 对象 1 被 变量a引用，对象1的引用计数器为1
    b = a  # 对象1 被变量b引用，对象1的引用计数器加1
    c = a  # 1对象1 被变量c引用，对象1的引用计数器加1
    del a  # 删除变量a，解除a对1的引用
    del b  # 删除变量b，解除b对1的引用
    print(a)  # 最终变量c仍然引用1

def test_dataloader():
    from dataloader import MusicDataset,CustomDataset
    import os
    from torch.utils.data import DataLoader
    data_dir = os.path.join(os.getcwd(), 'data' + os.sep)
    model_name = 'CP'
    data_mode = 'full'
    music_dataset = MusicDataset(data_dir, train_mode=model_name, data_mode=data_mode, is_train='train')
    print(len(music_dataset))
    train_num = len(music_dataset)

    music_dataloader = DataLoader(
        music_dataset, batch_size=10, shuffle=False, num_workers=0)
    for i, data in enumerate(music_dataloader):
        print(music_dataset._get_name(i))

def test_label():
    data_dir = os.path.join(os.getcwd(), 'data' + os.sep)
    dataA = glob.glob(data_dir +  'JCP_mixed\*' + '.npy')
    labelA = [(1.0, 0.0) for _ in range(len(dataA))]
    labelB = [(0.0, 1.0) for _ in range(len(dataA))]
    label = labelA + labelB
    a = label[len(labelA)] #列表拼接
    return a
if __name__ == '__main__':
    test_label()