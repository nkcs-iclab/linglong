import torch

# 从.pt文件中加载状态字典
state_dict = torch.load("/home/lidongwen/lidongwen/model/300M/V12.pt")

# state_dict = torch.load("/home/lidongwen/lidongwen/model/FT-Math23K-317M/models/vprompt-5-torch/E20L3.314032.pt")
parameter_names = list(state_dict.keys())
# print("Parameter names in the state dict:")
# for name in parameter_names:
#    print(name)


# 创建一个新的状态字典用于存储重命名后的键
new_state_dict = {}

for old_key, value in state_dict.items():
    if 'block' in old_key:
        # if "attn" in old_key:
        if "mlp" in old_key or 'attn' in old_key:
            if old_key.endswith("w"):
                new_key = old_key[:-1] + "weight"  # 根据需要进行重命名
            else:
                new_key = old_key[:-1] + "bias"  # 根据需要进行重命名
        else:
            new_key = old_key
    else:
        new_key = old_key
    # if 'block' in old_key:
    #     if '.1.' in old_key:
    #         print(old_key, new_key, old_key == new_key)
    # else:
    #     print(old_key, new_key, old_key == new_key)

    print(old_key, new_key, old_key == new_key)
    new_state_dict[new_key] = value
torch.save(new_state_dict, "/home/lidongwen/lidongwen/model/300M/V12-renamed-all.pt")
# torch.save(new_state_dict, "/home/lidongwen/lidongwen/model/FT-Math23K-317M/models/vprompt-5-torch/E20-renamed-all.pt")
