"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
这段代码演示了如何在小规模的Darcy-Flow数据集上训练一个张量化的Fourier神经算子TFNO。
"""

# %%
# 

#导入必须的库
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
#设备
device = 'cpu'

class H1Loss(object):
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L

# %%
# 加载数据集
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)
data_processor = data_processor.to(device)


# %%
# 创建TFNO模型
#实例化一个张量化的Fourier神经算子（TFNO）模型
# 设置了特定的参数，如模式数量、隐藏通道数、投影通道数、因子分解方法和秩。将模型移动到指定的设备（CPU）上
model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#优化器和调度器设置
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# 定义损失函数
#定义了L2损失和H1损失函数。选择H1损失作为训练损失，两者都用于评估。
#这些损失函数将用于训练和评估TFNO模型。
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%

#打印模型信息
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# 创建训练器
trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# 训练模型
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
# i) fits in just a few Mb of memory强调示例使用了小规模数据集和少量轮次，以便快速演示。
# ii) can be trained quickly on CPU在实际应用中，将使用更大的数据集、更高的分辨率和更多的训练轮次。
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[32].dataset
#绘制结果
fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.show()
#具体而言，模型在训练过程中学习调整其参数
#以最小化模型输出与已知近似解（可能是训练数据中提供的真实解）之间的差异
#通过这个过程，模型希望捕捉到更精确、更一般化的解
#在训练完成后，这个经过学习的模型可以用于对新的输入条件进行预测，生成更准确的解的近似值
#那么一开始的近似解可能很差，但是模型训练得到的解可能会更加接近真实解而不是一开始的近似解
#一开始的近似解可能很差，但是通过模型的训练，它有望学习到更加接近真实解的映射关系
#在训练的过程中，模型通过调整参数，不断减小预测结果与真实解之间的差异，逐渐改进模型的性能
#训练的目标是使模型能够对输入数据的变化做出更为准确的预测，即提高模型的泛化能力
#这使得模型在处理未见过的数据时能够表现得更好，而不仅仅是在训练数据上获得较好的拟合。
#因此，通过模型训练，可以期望得到一个对真实解更好的近似。

