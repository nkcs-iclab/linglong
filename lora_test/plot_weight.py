import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

sns.set(font_scale=1.5)


def read_att_new(file, new_file, para_name, key):
    para = {}
    state_dict = torch.load(file)
    print(state_dict.keys())
    weight = state_dict[para_name].cpu().numpy()
    new_state_dict = torch.load(new_file)
    # print(state_dict.keys())
    new_weight = new_state_dict[para_name].cpu().numpy()
    para["q"] = new_weight[:, :1024] - weight[:, :1024]
    para["k"] = new_weight[:, 1024:2048] - weight[:, 1024:2048]
    para["v"] = new_weight[:, 2048:] - weight[:, 2048:]
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


for i in range(24):
    para_name = f'transformer.blocks.{i}.attn.c_attn.w'
    key = 'k'
    new_file = f'/home/lidongwen/lidongwen/model/FT-Math23K-317M/models/vtest-torch/E10L5.034214.pt'
    file = f'/home/lidongwen/lidongwen/model/300M/V12.pt'
    data, vmin, vmax = read_att_new(file, new_file, para_name, key)
    plot(data, vmin, vmax)
    input()
