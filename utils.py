import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import yaml
import json
import math
import gc
import os

def device_assign(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print("Using device:{}".format(args.device))


def model_info_extr(args):
    path = args.config_path
    parts = path.split('/')
    model_name = parts[-2]
    config_version = parts[-1].split('.')[0].split('_')[-1]

    return model_name, config_version


def make_save_path(save_root, model_name, config_version):
    save_path = save_root + "/" + model_name + "/" + config_version

    if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    return save_path


def load_config(args):
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # copy config file
    save_path = make_save_path(config["save_root"], config["m_name"], config["co_ver"])
    shutil.copy(args.config_path, save_path)
    print(args.config_path, save_path)

    return config


def save_plot(history, save_path):
    y1_t = history.history["seq_acc"]
    y2_t = history.history["loss"]
    y1_v = history.history.get("val_seq_acc")
    y2_v = history.history.get("val_loss", [])

    upper_limit = 10
    y2_t = [min(val, upper_limit) for val in y2_t]
    y2_v = [min(val, upper_limit) for val in y2_v]

    fig = plt.figure(figsize=(40, 20))
    plt.rcParams["font.size"] = 25

    # データのプロット
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx() 
    ax1.plot(y1_t, marker="o", color='b', label="Acc_tra")
    ax2.plot(y2_t, marker="s", color='g', label="Los_tra")

    if y1_v and y2_v:
        ax1.plot(y1_v, marker="o", color='r', label="Acc_val")
        ax2.plot(y2_v, marker="s", color='y', label="Los_val")

    # X軸範囲の設定
    ax1.set_xlim(-5, 405)

    # X軸メモリの設定
    ax1.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])

    # Y軸範囲の設定
    ax1.set_ylim(-0.01, 1.01)
    ax2.set_ylim(-0.05, 10.05)

    # Y軸メモリの設定
    ax1.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,\
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax2.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,\
                    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10])

    #ラベルの設定
    ax1.set_title("Training Plot", pad=10)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Acc")
    ax2.set_ylabel("Los")
    ax1.legend(loc=4)
    ax2.legend(loc=5)

    ax1.grid()
    #ax2.grid()
    plt.tight_layout()
    #plt.show()

    #保存
    fig.savefig("{}/train_plot.png".format(save_path), format="png", dpi=300)