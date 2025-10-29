"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 导入库，TFNO是fno另一版本
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.training import CheckpointCallback
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
#规定设备
device = 'cpu'


# %%
# 将设备设置为CPU，并使用load_darcy_flow_small函数加载Darcy-Flow数据集。
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)


# %%
# 模型初始化，创建了一个TFNO模型，并将其移到指定的设备。打印了模型的参数数量。

model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#优化器和学习调度器
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# 创建L2和H1损失函数，将H1损失用作训练损失，同时定义了评估损失。
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
#创建了一个训练器对象，其中包含了模型、训练周期、设备、回调函数等设置。
trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  callbacks=[
                    CheckpointCallback(save_dir='./checkpoints',
                                       save_interval=10,
                                            save_optimizer=True,
                                            save_scheduler=True)
                        ],
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# 使用小规模的Darcy-Flow数据集训练模型。

trainer.train(train_loader=train_loader,
              test_loaders={},
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss)


# 创建了另一个训练器对象，用于从保存的检查点（在第10个周期时）恢复训练。
print(f"Resume from directory: {resume_from_dir}")


trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=data_processor,
                  callbacks=[
                    CheckpointCallback(save_dir='./new_checkpoints',
                                            resume_from_dir='./checkpoints/ep_10')
                        ],             
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)
#继续从第10个周期的检查点处训练模型。
trainer.train(train_loader=train_loader,
              test_loaders={},
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss)