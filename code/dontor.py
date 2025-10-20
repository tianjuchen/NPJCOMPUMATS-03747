import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepONet_Model(nn.Module):
    def __init__(self, Par):
        super(DeepONet_Model, self).__init__()
        # 设置随机种子以保证可复现性
        np.random.seed(23)
        torch.manual_seed(23)
        
        # 模型超参数
        self.latent_dim = 5  # 特征对齐的潜在空间维度
        self.m = 196         # 用于组合分支网络和主干网络输出的"基"组件数量
        self.Par = Par       # 外部参数（如归一化均值/标准差、输入通道数等）
        
        # 记录训练指标
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        
        # 优化器（学习率与原版本保持一致）
        self.lr = 1e-4
        
        # 构建分支网络（处理输入函数，基于CNN）
        self.branch_net = self.build_branch_net()
        # 构建主干网络（处理查询位置，基于全连接层）
        self.trunk_net = self.build_trunk_net()
        
        # 可训练的缩放因子（与原版本一致）
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def build_branch_net(self):
        """构建分支网络：处理2D输入函数（如图像）"""
        return nn.Sequential(
            # 卷积层1：32个3x3滤波器，输入形状为[channels, 14, 14]（PyTorch使用通道优先格式）
            nn.Conv2d(
                in_channels=self.Par['n_channels'],
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0  # 无填充，14x14 → 12x12
            ),
            nn.Activation(F.sin),  # 正弦激活函数（替代ReLU）
            nn.BatchNorm2d(32),    # 批归一化
            
            # 卷积层2：16个3x3滤波器
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0  # 12x12 → 10x10
            ),
            nn.Activation(F.sin),
            nn.BatchNorm2d(16),
            
            # 卷积层3：16个3x3滤波器
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0  # 10x10 → 8x8
            ),
            nn.Activation(F.sin),
            nn.BatchNorm2d(16),
            
            # 展平特征图
            nn.Flatten(),
            # 全连接层：投影到m*latent_dim维度
            nn.Linear(16 * 8 * 8, self.m * self.latent_dim)  # 16通道×8×8特征图 → 输出维度
        )

    def build_trunk_net(self):
        """构建主干网络：处理查询位置（如空间/时间坐标）"""
        return nn.Sequential(
            # 全连接层1：100个单元
            nn.Linear(1, 100),  # 输入为1D位置坐标
            nn.Activation(F.sin),
            
            # 全连接层2：100个单元
            nn.Linear(100, 100),
            nn.Activation(F.sin),
            
            # 全连接层：投影到m*latent_dim维度
            nn.Linear(100, self.m * self.latent_dim)
        )

    def forward(self, X_func, X_loc):
        """前向传播：组合分支网络和主干网络的输出"""
        # X_func：输入函数，形状为[batch_size, n_channels, 14, 14]（PyTorch通道优先）
        # X_loc：查询位置，形状为[n_locations, 1]
        
        # 1. 处理输入函数（分支网络）
        # 归一化输入函数（使用预计算的均值和标准差）
        y_func = (X_func - self.Par['mean']) / self.Par['std']
        # 传入分支网络
        y_func = self.branch_net(y_func)  # 输出形状：[batch_size, m*latent_dim]
        
        # 2. 处理查询位置（主干网络）
        # 位置缩放：将[0,1]范围映射到[-5,5]（与原版本一致）
        y_loc = 10 * (X_loc - 0.5)
        # 传入主干网络
        y_loc = self.trunk_net(y_loc)  # 输出形状：[n_locations, m*latent_dim]
        
        # 3. 特征对齐与组合
        # 重塑为[m, latent_dim]维度（便于点积计算）
        y_func = y_func.view(-1, self.m, self.latent_dim)  # [batch_size, m, latent_dim]
        y_loc = y_loc.view(-1, self.m, self.latent_dim)    # [n_locations, m, latent_dim]
        
        # 张量收缩（tensor contraction）：在latent_dim维度上计算点积，组合特征
        # 等价于原TensorFlow的einsum('ijk,pjk->ipj')
        Y = torch.einsum('ijk,pjk->ipj', y_func, y_loc)  # 输出：[batch_size, n_locations, m]
        
        return Y

    def loss_fn(self, y_pred, y_train):
        """损失函数：均方误差（MSE）"""
        return torch.mean(torch.square(y_pred - y_train))

    def get_optimizer(self):
        """获取优化器（便于外部调用）"""
        return torch.optim.Adam(self.parameters(), lr=self.lr)