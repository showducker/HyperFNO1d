"""
Training a SFNO on the spherical Shallow Water equations
==========================================================

In this example, we demonstrate how to use the small Spherical Shallow Water Equations example we ship with the package
to train a Spherical Fourier-Neural Operator
这段代码是一个示例,演示了如何使用神经运算符Neural Operator来训练一个基于球面浅水方程Spherical Shallow Water Equations的SFNO模型。
"""

# %%
# 

#导入必要的库
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import SFNO
from neuralop import Trainer
from neuralop.datasets import load_spherical_swe
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
#设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
# 加载数据集
train_loader, test_loaders = load_spherical_swe(n_train=200, batch_size=4, train_resolution=(32, 64),
                                                test_resolutions=[(32, 64), (64, 128)], n_tests=[50, 50], test_batch_sizes=[10, 10],)


# %%
# 创建模型

model = SFNO(n_modes=(32, 32), in_channels=3, out_channels=3, hidden_channels=32, projection_channels=64, factorization='dense')
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#创建优化器和学习率调度器
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-4, 
                                weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# 定义损失函数
l2loss = LpLoss(d=2, p=2, reduce_dims=(0,1))
# h1loss = H1Loss(d=2, reduce_dims=(0,1))

train_loss = l2loss
eval_losses={'l2': l2loss} #'h1': h1loss, 


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# 创建Trainer
trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# 训练模型
#通过优化模型的参数，使得模型在训练数据上产生的输出尽可能接近真实的解。
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs
#可视化结果，画图
fig = plt.figure(figsize=(7, 7))
for index, resolution in enumerate([(32, 64), (64, 128)]):
    test_samples = test_loaders[resolution].dataset
    data = test_samples[0]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y'][0, ...].numpy()
    # Model prediction
    x_in = x.unsqueeze(0).to(device)
    out = model(x_in).squeeze()[0, ...].detach().cpu().numpy()
    x = x[0, ...].detach().numpy()

    ax = fig.add_subplot(2, 3, index*3 + 1)
    ax.imshow(x)
    ax.set_title(f'Input x {resolution}')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index*3 + 2)
    ax.imshow(y)
    ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index*3 + 3)
    ax.imshow(out)
    ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.show()
#意思是这段代码没有具体求解方程的代码
# 只是有个已知的近似解，然后学习这个解的映射
# 在训练过程中得到比这个近似解还好的解或者叫精确解

#具体而言，模型在训练过程中学习调整其参数
# 以最小化模型输出与已知近似解（可能是训练数据中提供的真实解）之间的差异
# 通过这个过程，模型希望捕捉到更精确、更一般化的球面浅水方程的解
# 在训练完成后，这个经过学习的模型可以用于对新的输入条件进行预测
# 生成更准确的球面浅水方程的解的近似值。