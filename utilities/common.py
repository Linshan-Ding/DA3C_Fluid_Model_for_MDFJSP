"""
轻量级通用工具：仅依赖标准库。
环境与算法代码从这里导入 MyError / AddData / 路径与种子工具，
避免拖入 matplotlib / openpyxl / pandas 等绘图依赖（见 Utility_Class.py）。
"""
import csv
import os
import random
from pathlib import Path

# 仓库根目录（utilities/ 的上一级）
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'
DATA_GENERALIZATION_DIR = REPO_ROOT / 'data_generalization'
RESULT_DIR = REPO_ROOT / 'result'
CSV_DIR = RESULT_DIR / 'csv'
LATEX_DIR = RESULT_DIR / 'latex'
MODEL_DIR = RESULT_DIR / 'models'

# 论文测试用的 27 个基准实例
INSTANCES_27 = ['DDT{}_M{}_S{}'.format(ddt, m, s)
                for ddt in ('0.5', '1.0', '1.5')
                for m in (10, 15, 20)
                for s in (1, 3, 5)]


class MyError(Exception):
    """自定义异常"""
    def __init__(self, error_information):
        super().__init__(self)
        self.error_information = error_information

    def __str__(self):
        return self.error_information


class AddData():
    """逐行追加写入 csv 文件"""
    def __init__(self, path_file_name):
        self.file_name = str(path_file_name)
        os.makedirs(os.path.dirname(self.file_name) or '.', exist_ok=True)

    def add_data(self, data):
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)


def write_csv(path, header, rows):
    """整体覆盖写入 csv 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path):
    """读取 csv，返回 (header, rows) 均为字符串列表"""
    with open(path, newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
    return rows[0], rows[1:]


def set_global_seed(seed):
    """统一控制 random / numpy / torch 三处随机源（后两者可选依赖）"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def parse_instance_name(file_name):
    """'DDT1.0_M15_S3' -> (1.0, 15, 3)"""
    ddt_part, m_part, s_part = file_name.split('_')
    return float(ddt_part[3:]), int(m_part[1:]), int(s_part[1:])
