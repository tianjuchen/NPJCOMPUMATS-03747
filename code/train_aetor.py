import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
import os

from aetor import AE  # 假设之前的AE类已经定义

def train(model, x, optimizer, device):
    """
    训练一个batch
    """
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    y_pred = model(x)
    
    # 计算损失
    loss = model.loss(y_pred, x)[0]
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    # 设置随机种子
    torch.manual_seed(23)
    np.random.seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(23)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据集
    d = np.load('data/train_128_128.npz')
    x_train = d['X_func']
    
    # 数据预处理和 reshaping
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    x_train = np.reshape(x_train, (-1, 100, 128, 128, 1))
    # x_train = x_train[:, 10:]  # 注释掉的代码
    x_train = np.reshape(x_train, (-1, 128, 128, 1))
    num_samples = x_train.shape[0]
    
    # 数据归一化
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    print('x_train shape: ', x_train.shape)
    
    # 加载测试集
    d = np.load('data/test_128_128.npz')
    x_test = d['X_func']
    
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    x_test = np.reshape(x_test, (-1, 100, 128, 128, 1))
    x_test = np.reshape(x_test, (-1, 128, 128, 1))
    
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
    print('x_test shape: ', x_test.shape)
    
    # 转换为PyTorch张量并调整维度顺序 (channel-first)
    x_train_tensor = torch.FloatTensor(x_train).permute(0, 3, 1, 2)  # [N, 1, 128, 128]
    x_test_tensor = torch.FloatTensor(x_test).permute(0, 3, 1, 2)    # [N, 1, 128, 128]
    
    # 创建数据加载器
    batch_size = 1
    train_dataset = TensorDataset(x_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 保存地址
    address = 'saved_models/ae_models'
    os.makedirs(address, exist_ok=True)
    
    # 创建模型
    model = AE().to(device)
    print('Model created')
    
    # 训练参数
    n_epochs = 12
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 记录训练过程
    index_list = []
    train_loss_list = []
    val_loss_list = []
    
    begin_time = time.time()
    print('Training Begins')
    
    for epoch in range(n_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        # 训练阶段
        for batch_data in train_loader:
            x_batch = batch_data[0].to(device)
            
            loss = train(model, x_batch, optimizer, device)
            epoch_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # 验证和保存
        if epoch % 1 == 0:
            # 保存模型权重
            torch.save(model.state_dict(), f"{address}/model_{epoch}.pth")
            
            # 验证损失
            model.eval()
            with torch.no_grad():
                x_test_batch = x_test_tensor[:100].to(device)  # 使用部分测试数据计算验证损失
                y_pred = model(x_test_batch)
                val_loss = nn.functional.mse_loss(y_pred, x_test_batch).item()
            
            elapsed_time = int(time.time() - begin_time)
            print(f"epoch: {epoch}, Train Loss: {avg_train_loss:.3e}, "
                  f"Val Loss: {val_loss:.3e}, elapsed time: {elapsed_time}s")
            
            index_list.append(epoch)
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(val_loss)
    
    print('Training complete')
    
    # 收敛曲线图
    plt.figure(figsize=(10, 7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig(f"{address}/convergence.png", dpi=800)
    plt.close()
    
    # 找到最佳模型
    best_model_number = index_list[np.argmin(val_loss_list)]
    print('Best autoencoder model: ', best_model_number)
    
    # 保存最佳模型编号
    np.save(f'{address}/best_ae_model_number.npy', best_model_number)
    
    print('--------Complete--------')

if __name__ == "__main__":
    main()