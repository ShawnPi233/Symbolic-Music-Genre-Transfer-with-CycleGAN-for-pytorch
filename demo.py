import torch
beta = torch.linspace(0.001,0.2,100)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha,dim=0).to('cuda')
def f()->int:
    global alpha_bar
    return alpha_bar
b = f()
print(b)