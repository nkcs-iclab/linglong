import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

sns.set(font_scale=1.5)


def read_att(file, para_name, key):
    para = {}
    state_dict = torch.load(file)
    print(state_dict.keys())
    weight = state_dict[para_name].numpy()
    para["q"] = weight[:, :1024]
    para["k"] = weight[:, 1024:2048]
    para["v"] = weight[:, 2048:]
    data = para[key]
    vmin = min(data.flatten())
    vmax = max(data.flatten())
    return data, vmin, vmax


def read_lora_att(file, para_name, key, r):
    state_dict = torch.load(file)
    loraA = state_dict[para_name + ".lora_A"].cpu().numpy()
    loraB = state_dict[para_name + ".lora_B"].cpu().numpy()
    para = {}
    para["q_A"] = loraA[:r, :]
    para["k_A"] = loraA[r:2 * r, :]
    para["v_A"] = loraA[2 * r:, :]
    para["q_B"] = loraB[:1024, :]
    para["k_B"] = loraB[1024:2048, :]
    para["v_B"] = loraB[2048:, :]
    para["q"] = np.dot(np.transpose(para["q_A"]), np.transpose(para["q_B"]))
    para["k"] = np.dot(np.transpose(para["k_A"]), np.transpose(para["k_B"]))
    para["v"] = np.dot(np.transpose(para["v_A"]), np.transpose(para["v_B"]))
    # print(q.shape, k.shape, v.shape)
    para["qk"] = np.dot(para["q"], np.transpose(para["k"]))
    para["qkv"] = np.dot(para["qk"], para["v"])
    data = para[key]
    vmin = min(data.flatten())
    vmax = max(data.flatten())
    return data, vmin, vmax


def plot(data, vmin, vmax):
    sns.set_context({"figure.figsize": (8, 8)})
    # sns.distplot(data.flatten(), kde=True)
    # plt.show()
    sns.heatmap(data=data, vmin=vmin, vmax=vmax, square=True, cmap="RdBu_r")
    plt.show()


# file = '/home/lidongwen/lidongwen/model/300M/V12-renamed-all.pt'
# for i in range(24):
#     para_name = f'transformer.blocks.{i}.attn.c_attn.weight'
#     key = 'k'
#     data, vmin, vmax = read_att(file, para_name, key)
#     plot(data, vmin, vmax)

r = 16
for i in range(24):
    para_name = f'transformer.blocks.{i}.attn.c_attn'
    key = 'k'
    file = f'/home/lidongwen/lidongwen/model/FT-Math23K-317M/models/vLoRA-{r}-all-torch/E9L0.961448.pt'
    data, vmin, vmax = read_lora_att(file, para_name, key, r)
    plot(data, vmin, vmax)
    # file = '/home/lidongwen/lidongwen/model/FT-AdGen-317M/models/vLoRA-8-all-torch/E1L1.605822.pt'
    # data, vmin, vmax = read_lora_att(file, para_name, key)
    # plot(data, vmin, vmax)
    input()
