import torch
import numpy as np
target = [5,4,6,8,7,10]
print(np.array(target))
target = torch.from_numpy(np.array(target)-1).long()
mask = torch.arange(target.max().item())[None,:] < target[:,None]
mask =mask.float()
print(torch.arange((target.max().item())))
target = target.contiguous().view(-1,1)
mask = mask.contiguous().view(-1, 54)
change = torch.nn.Linear(in_features=54,out_features=6)
print(target.shape)
print(mask.shape)


input = change(mask)
input = input.contiguous().view(-1,1)

result = input.gather(1,target)*mask
print(result.shape)
