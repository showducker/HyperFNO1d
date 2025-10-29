"""
A simple Darcy-Flow dataset
===========================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
"""

# %%
# Import the library
# ------------------
# We first import our `neuralop` library and required dependencies.
#导入库
import matplotlib.pyplot as plt
from neuralop.datasets import load_darcy_flow_small

# %%
# 加载数据集
# ----------------
# Training samples are 16x16 and we load testing samples at both 
# 16x16 and 32x32 (to test resolution invariance).
#代码使用 load_darcy_flow_small 函数加载 Darcy-Flow 数据集。它指定了一些参数，如训练样本数量 (n_train)、训练批次大小 (batch_size)，以及在不同分辨率下测试样本的详细信息。
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=100, batch_size=4, 
        test_resolutions=[16, 32], n_tests=[50, 50], test_batch_sizes=[4, 2],
        )

train_dataset = train_loader.dataset

# %%
# 数据可视化
# --------------------

#代码打印了在不同分辨率下测试样本的形状信息以及一个训练样本的形状信息。
for res, test_loader in test_loaders.items():
    print('res')
    test_data = train_dataset[0]
    x = test_data['x']
    y = test_data['y']
    #测试
    # print(y)
    # print(x)
    # print(x.shape)
    # print(y.shape)
    print(f'Testing samples for res {res} have shape {x.shape[1:]}')


data = train_dataset[0]
x = data['x']
y = data['y']
# print(y)
# print(x)
# print(x.shape)
# print(y.shape)
print(f'Training sample have shape {x.shape[1:]}')


# 单个数据样本可视化
#代码随后可视化了一个单独的训练样本，展示了输入数据的不同方面。
#它从训练样本中提取了输入数据 (x 和 y)，并使用 Matplotlib 显示它们。
index = 0

data = train_dataset[index]
data = data_processor.preprocess(data, batched=False)
x = data['x']
y = data['y']
# print(y)
# print(x)
# print(x.shape)
# print(y.shape)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(2, 2, 1)
ax.imshow(x[0], cmap='gray')
ax.set_title('input x')
ax = fig.add_subplot(2, 2, 2)
ax.imshow(y.squeeze())
ax.set_title('input y')
ax = fig.add_subplot(2, 2, 3)
ax.imshow(x[1])
ax.set_title('x: 1st pos embedding')
ax = fig.add_subplot(2, 2, 4)
ax.imshow(x[2])
ax.set_title('x: 2nd pos embedding')
fig.suptitle('Visualizing one input sample', y=0.98)
plt.tight_layout()
plt.show()
