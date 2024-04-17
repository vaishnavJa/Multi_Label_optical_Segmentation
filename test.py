import torch
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(0, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

pre_a = torch.arange(1,10)
a = torch.cumprod(pre_a,0)
a = F.pad(a[:-1], (1, 0), value=1.0)
t = torch.randint(0,10,size=(3,))
x = torch.randint(0,5,size=(3,2,3))
y= extract(a,t,x.shape) * x
x = 1