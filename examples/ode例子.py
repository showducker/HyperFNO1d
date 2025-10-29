#导入库
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from neuralop.models import TFNO1d
# from FCNN import fcnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

#设备选择，检查CUDA是否可用，并根据结果选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据准备，生成用于训练的数据 X_Train，其中包括空间坐标k和时间坐标nt
k = torch.linspace(0, 1, 21).reshape((-1, 1, 1))
k = k.repeat(1, 1, 201)
nt = torch.linspace(0, 1, 201).reshape(1, 1, -1).requires_grad_()
nt = nt.repeat(21, 1, 1)
X_Train = torch.cat((k, nt), dim=1)

#真实函数和梯度的计算，定义真实函数 f 并计算真实梯度 dydx_true。
def f(x):
    y = 0.5 + 0.5 * x[:, 0:1, :] * torch.cos(x[:, 1:2, :] * 2 * np.pi)
    return y
y_true = f(X_Train)
dydx_true = -0.5 * X_Train[:, 0:1, :] * 2 * np.pi * torch.sin(X_Train[:, 1:2, :] * 2 * np.pi)

#模型定义
torch.manual_seed(1)
model = TFNO1d(n_modes_height=12, 
               hidden_channels=16,
               in_channels=2,
               out_channels=1).to(device)

#设置训练的迭代次数、学习率、优化器和学习率调度器
iteration = 0
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for param_group in optimizer.param_groups:
    param_group['initial_lr'] = 1e-3

scheduler = ReduceLROnPlateau(optimizer,
                              factor=0.5,
                              patience=50,
                              min_lr=1e-6)

X_Train = X_Train.to(device)
y_true = y_true.to(device)
h = 1e-5
start = time.time()

# 初始化损失记录
losses = []

#训练模型，使用梯度下降优化器训练模型，打印迭代次数、损失和学习率
while iteration < 5000:
    y_pred = model(X_Train)
    dydx_pred = (model(X_Train + h) - model(X_Train - h)) / (2 * h)
    res1 = dydx_pred - dydx_true

    optimizer.zero_grad()

    # L2 损失（均方误差）：用于比较模型对 y 的预测与真实值之间的差异
    loss_y = torch.mean(torch.square(y_pred - y_true))
    # L2 损失（均方误差）：用于比较模型对 dy/dx 的预测与真实值之间的差异
    loss_dydx = torch.mean(torch.square(dydx_pred - dydx_true))
    # 最后时刻的预测值与目标值之间的均方误差
    loss_last_point = torch.mean(torch.square(y_pred[:, 0, -1] - 1.0))
    # 总损失，可以根据具体情况调整各部分的权重
    loss = loss_y + loss_dydx + loss_last_point

    # 分别计算各个损失项的梯度
    loss_y.backward(retain_graph=True)
    loss_dydx.backward(retain_graph=True)
    loss_last_point.backward()

    # 更新参数
    optimizer.step()
    scheduler.step(loss.item())

    # 记录损失值
    losses.append(loss.item())

    iteration += 1
    print(f"iteration {iteration} | loss: {loss.item()} | lr: {optimizer.param_groups[0]['lr']}")

print(f"Total Training Time: {time.time() - start} seconds.")

#将最终得到的预测结果和真实数据移回 CPU，并将梯度信息从张量中分离
y_pred = model(X_Train)
dydx_pred = (model(X_Train + h) - model(X_Train - h)) / (2 * h)
y_pred = y_pred.detach().cpu()
dydx_true = dydx_true.detach().cpu()
X_Train = X_Train.detach().cpu()
y_true = y_true.detach().cpu()
dydx_pred = dydx_pred.detach().cpu()

#绘制模型的最终预测结果和真实值
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_pred.squeeze()[-1, :].detach().cpu(), label='HyperFNO prediction')
ax.plot(y_true.squeeze()[-1, :].detach().cpu(), label='True', alpha=0.9, ls='-.')
ax.legend()
ax.set_title('$y = 0.5 + 0.5 k \\times \cos(2 \pi x)$')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dydx_pred.squeeze()[-1, :].detach().cpu(), label='HyperFNO prediction')
ax.plot(dydx_true.squeeze()[-1, :].detach().cpu(), label='True', alpha=0.9, ls='-.')
ax.set_title('$dy/dx =  - 0.5 k \\times 2 \pi \sin(2 \pi x)$')
ax.legend()
plt.show()

# 生成整个输入空间的网格
k_values = torch.linspace(0, 1, 100).reshape((-1, 1, 1)).to(device)
nt_values = torch.linspace(0, 1, 100).reshape((1, 1, -1)).to(device)
grid = torch.cat((k_values.repeat(1, 1, 100), nt_values.repeat(100, 1, 1)), dim=1)

# 将网格输入到模型中进行预测
model.eval()
with torch.no_grad():
    predictions = model(grid)

# 将预测结果和真实值移回 CPU
predictions = predictions.cpu().numpy()

# 绘制预测结果
plt.figure(figsize=(10, 8))

# 绘制真实函数
plt.subplot(1, 2, 1)
plt.imshow(f(grid).squeeze(), extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
plt.title('True Function')
plt.xlabel('k')
plt.ylabel('nt')

# 绘制模型预测的 y
plt.subplot(1, 2, 2)
plt.imshow(predictions.squeeze(), extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
plt.title('Model Prediction for y')
plt.xlabel('k')
plt.ylabel('nt')

plt.show()

# 绘制损失值随迭代次数变化的图
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
