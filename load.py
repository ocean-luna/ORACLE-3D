import torch

# 加载模型
model_path = '/volumes/jointmodel/wyr/code/sam_clip3d/output/foundationTrans/nuscenes/210524-0347/model.pt'

model = torch.load(model_path)

# 打印模型结构
print(model)