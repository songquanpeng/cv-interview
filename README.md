# 计算机视觉算法岗常见算法的实现
> What I cannot create, I do not understand.

## 内容
1. 常见网络层的实现（[卷积层](./layers/Conv2d.py)，[线性层](./layers/Linear.py)）。
2. 常见损失函数（[各种形式的对抗损失](./losses/adversarial_loss.py)），距离度量，散度以及相似性函数的实现。
3. 常见指标的实现（[Inception Score](./metrics/IS.py), [Fréchet Inception Distance](./metrics/FID.py)）。
4. 常见机器学习算法的实现（[多层感知机](./ml_models/MLP.py)）。
5. 经典网络的实现（[DCGAN](./dl_models/DCGAN.py), [VGG16](./dl_models/VGG16.py), [ResNet](./dl_models/ResNet.py)）。
6. 常见归一化方法的实现（[BatchNorm](./normalizations/BatchNorm.py)）。
7. 常见优化器的实现（[SGD](./optimizers/SGD.py), [Momentum](./optimizers/Momentum.py), [Adagrad](./optimizers/Adagrad.py), [RMSprop](./optimizers/RMSprop.py), [Adam](./optimizers/Adam.py)）。
8. 常见正则化方法的实现（[R1 正则化](./regularization/r1_regularization.py)）。
9. 常见可视化方法的实现。
10. 常见激活函数的实现。
11. 常见图像处理的实现。
12. 常见的权重初始化方法的实现。

## 相关项目
+ PyTorch 项目模板：https://github.com/songquanpeng/pytorch-template
+ PyTorch + Flask 项目部署模板：https://github.com/songquanpeng/pytorch-deployment