import torch

# 从.pt文件中加载状态字典
state_dict = torch.load("/home/lidongwen/lidongwen/model/300M/V12.pt")
new_state_dict = {}

for key, value in state_dict.items():
    new_value = torch.ones_like(value)
    new_state_dict[key] = new_value

torch.save(new_state_dict, "/home/lidongwen/lidongwen/model/300M/one.pt")
