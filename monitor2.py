import threading
import time
import subprocess
import torch
import time


# 模拟在特定GPU上运行的模型计算任务
import torch

def model_work(gpu_id):
    # 设置当前脚本使用的GPU
    torch.cuda.set_device(gpu_id)
    # 创建大型随机矩阵
    size = 5000 # 这指定了矩阵的规模
    a = torch.randn(size, size, device=f'cuda:{gpu_id}')
    b = torch.randn(size, size, device=f'cuda:{gpu_id}')
    # 执行密集型的矩阵乘法。这将充分使用GPU资源进行大量的计算
    for _ in range(800):
        torch.matmul(a, b)



# 获取当前显卡利用率的函数
def get_current_gpu_utilization(gpu_id):
    result = subprocess.run(
        ['nvidia-smi', f'--id={gpu_id}', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
        capture_output=True,
        text=True
    )
    utilization_str = result.stdout.strip()
    return int(utilization_str.replace(' %', ''))


# 检查单个GPU的利用率并根据需要执行工作
def monitor_and_optimize_gpu_utilization(gpu_id):
    while True:
        utilization = get_current_gpu_utilization(gpu_id)
        if utilization < 50:  # 假设阈值为 50%
            model_work(gpu_id)
        time.sleep(1)


# 为每个GPU启动一个监控和优化利用率的线程
def optimize_multi_gpu_utilization():
    gpu_count = torch.cuda.device_count()  # 获取GPU数量
    threads = []
    for gpu_id in range(gpu_count):
        thread = threading.Thread(target=monitor_and_optimize_gpu_utilization, args=(gpu_id,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()  # 等待所有线程完成，这里可以根据实际情况决定是否需要


optimize_multi_gpu_utilization()
