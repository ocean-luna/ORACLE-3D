import threading
import time
import GPUtil
import subprocess
import re
import warnings

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torchvision.models import alexnet  # 从 torchvision 导入 alexnet
import numpy as np
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = alexnet()  # 使用 alexnet

    def forward(self, x):
        return self.net(x)

device_count = torch.cuda.device_count()
# 将模型复制到多个GPU
model = SimpleModel()
model = DataParallel(model,device_ids=list(range(device_count)))  # 使用0到device_count-1号GPU
model.cuda()
# 创建一个假设的大输入，使得GPU有足够的工作量
input_size = (64, 3, 1024, 1024) # 这里你可以根据需要调整数据的尺寸，224x224是AlexNet期望的输入尺寸
inputs = torch.randn(input_size).cuda()

# 优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def launch_additional_gpu_task():
    optimizer.zero_grad()
    outputs = model(inputs)
    labels = torch.randint(0, 1000, (64,)).cuda()  # 随机假设的标签
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
def monitor_gpu_and_execute_tasks():
    while True:
        gpu_utilization = get_all_gpu_utilization() # 自定义函数，获取GPU利用率
        if gpu_utilization[0] < 50:
            launch_additional_gpu_task() # 自定义函数，启动额外的GPU任务


def get_all_gpu_utilization():
    # 执行nvidia-smi命令，获取所有GPU的状态
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
        capture_output=True,
        text=True
    )
    # 解析输出结果，获取所有利用率数字
    utilization_str = result.stdout.strip()
    # 利用正则表达式匹配，并转换成整数列表
    utilizations = [int(x.replace(' %', '')) for x in utilization_str.split('\n')]
    return utilizations




if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message=".*is_namedtuple is deprecated.*")
    monitor_thread = threading.Thread(target=monitor_gpu_and_execute_tasks)
    monitor_thread.start()