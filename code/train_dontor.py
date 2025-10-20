import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os

from aetor import AE  # 假设之前的AE类已经定义
from dontor import DeepONet_Model  # 假设已经实现了PyTorch版本的DeepONet

import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

def preprocess(x):
    X_func = np.reshape(x, (-1, 100, 14, 14))
    index = list(range(10, 90))
    X_func = X_func[:, list(range(10, 90))]
    X_func = np.transpose(X_func, axes=[0, 2, 3, 1])
    print(X_func.shape)

    X_loc = np.array(index) / 100
    X_loc = X_loc[:, None]
    print(X_loc.shape)

    y = np.reshape(x, (-1, 100, 196))
    y = y[:, index]
    print(y.shape)

    return X_func, X_loc, y

def tensor(x, device):
    return torch.FloatTensor(x).to(device)

def train(don_model, X_func, X_loc, y, optimizer):
    """
    训练一个batch的DeepONet
    """
    don_model.train()
    optimizer.zero_grad()
    
    # 前向传播
    y_hat = don_model(X_func, X_loc)
    
    # 计算损失
    loss = don_model.loss(y_hat, y)[0]
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item()

def error_metric(true, pred):
    """
    计算相对L2误差
    """
    # true - [samples, time steps, 128, 128]
    # pred - [samples, time steps, 128, 128]
    pred = np.reshape(pred, (-1, 90, 128, 128))
    num = np.abs(true - pred) ** 2  # [samples, time steps, 128, 128]
    num = np.sum(num)  # [samples, time steps]
    den = np.abs(true) ** 2
    den = np.sum(den)

    return num / den

def show_error(don_model, ae_model, X_func, X_loc, pf_true, device):
    """
    显示模型误差
    """
    don_model.eval()
    ae_model.eval()
    
    with torch.no_grad():
        y_pred = don_model(X_func, X_loc)
        y_pred = y_pred.cpu().numpy().reshape(-1, ae_model.latent_dim)
        
        # 使用AE解码器重建图像
        pf_pred = ae_model.decode(torch.FloatTensor(y_pred).to(device))
        pf_pred = pf_pred.cpu().numpy()
        
        error = error_metric(pf_true, pf_pred)
        print('L2 norm of relative error: ', error)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(23)
    np.random.seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(23)
    
    Par = {}
    
    # 加载数据集
    train_dataset = np.load('data/train_ld.npz')['X_func']
    test_dataset = np.load('data/test_ld.npz')['X_func']
    
    Par['address'] = 'saved_models/don_models'
    os.makedirs(Par['address'], exist_ok=True)
    
    print(Par['address'])
    print('------\n')
    
    # 数据预处理
    X_func_train, X_loc_train, y_train = preprocess(train_dataset)
    X_func_test, X_loc_test, y_test = preprocess(test_dataset)
    Par['n_channels'] = X_func_train.shape[-1]
    
    print('X_func_train: ', X_func_train.shape, '\nX_loc_train: ', X_loc_train.shape, '\ny_train: ', y_train.shape)
    print('X_func_test: ', X_func_test.shape, '\nX_loc_test: ', X_loc_test.shape, '\ny_test: ', y_test.shape)
    
    Par['mean'] = np.mean(X_func_train)
    Par['std'] = np.std(X_func_train)
    
    print('mean: ', Par['mean'])
    print('std : ', Par['std'])
    
    # 转换为PyTorch张量并移动到设备
    X_func_train_tensor = tensor(X_func_train, device)
    X_loc_train_tensor = tensor(X_loc_train, device)
    y_train_tensor = tensor(y_train, device)
    
    X_func_test_tensor = tensor(X_func_test, device)
    X_loc_test_tensor = tensor(X_loc_test, device)
    y_test_tensor = tensor(y_test, device)
    
    # 创建模型
    don_model = DeepONet_Model(Par).to(device)
    
    # 训练参数
    n_epochs = 12
    batch_size = 1
    
    # 优化器
    optimizer = optim.Adam(don_model.parameters(), lr=1e-4)
    
    print("DeepONet Training Begins")
    begin_time = time.time()
    
    # 记录训练过程
    index_list = []
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(n_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        # 训练循环
        for start in range(0, X_func_train_tensor.shape[0], batch_size):
            end = start + batch_size
            if end > X_func_train_tensor.shape[0]:
                end = X_func_train_tensor.shape[0]
            
            loss = train(
                don_model,
                X_func_train_tensor[start:end],
                X_loc_train_tensor,
                y_train_tensor[start:end],
                optimizer
            )
            
            epoch_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # 验证和保存
        if epoch % 1 == 0:
            # 保存模型
            torch.save(don_model.state_dict(), f"{Par['address']}/model_{epoch}.pth")
            
            # 验证损失
            don_model.eval()
            with torch.no_grad():
                y_hat = don_model(X_func_test_tensor, X_loc_test_tensor)
                val_loss = nn.functional.mse_loss(y_hat, y_test_tensor).item()
            
            elapsed_time = int(time.time() - begin_time)
            print(f"epoch: {epoch}, Train Loss: {avg_train_loss:.3e}, "
                  f"Val Loss: {val_loss:.3e}, elapsed time: {elapsed_time}s")
            
            index_list.append(epoch)
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(val_loss)
    
    # 收敛曲线图
    np.savez(f"{Par['address']}/convergence_data.npz", 
             index_list=index_list, 
             train_loss_list=train_loss_list, 
             val_loss_list=val_loss_list)
    
    plt.figure(figsize=(10, 7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig(f"{Par['address']}/convergence.png", dpi=800)
    plt.close()
    
    # 加载最佳模型并评估
    if True:
        # 加载AE模型
        ae_model = AE().to(device)
        ae_model_number = np.load('saved_models/ae_models/best_ae_model_number.npy')
        ae_model_address = f"saved_models/ae_models/model_{ae_model_number}.pth"
        ae_model.load_state_dict(torch.load(ae_model_address, map_location=device))
        
        # 加载最佳DeepONet模型
        don_model = DeepONet_Model(Par).to(device)
        don_model_number = index_list[np.argmin(val_loss_list)]
        np.save('data/best_don_model_number.npy', don_model_number)
        don_model_address = f"{Par['address']}/model_{don_model_number}.pth"
        don_model.load_state_dict(torch.load(don_model_address, map_location=device))
        
        print('Best DeepONet model: ', don_model_number)
        
        n_samples = 20
        
        # 加载真实物理场数据用于误差计算
        pf_true = np.load('data/train_128_128.npz')['X_func'].astype(np.float32)
        pf_true = (pf_true[:10000] - np.min(pf_true)) / (np.max(pf_true) - np.min(pf_true))
        pf_true = np.reshape(pf_true, (-1, 100, 128, 128))
        pf_true_train = pf_true[:n_samples, 10:]
        
        pf_true = np.load('data/test_128_128.npz')['X_func'].astype(np.float32)
        pf_true = (pf_true[:10000] - np.min(pf_true)) / (np.max(pf_true) - np.min(pf_true))
        pf_true = np.reshape(pf_true, (-1, 100, 128, 128))
        pf_true_test = pf_true[:n_samples, 10:]
        
        X_loc = np.linspace(0, 1, 100)[10:][:, None]
        X_loc_tensor = tensor(X_loc, device)
        
        print('Train Dataset')
        show_error(
            don_model, 
            ae_model, 
            X_func_train_tensor[:n_samples], 
            X_loc_tensor, 
            pf_true_train, 
            device
        )
        
        print('Test Dataset')
        show_error(
            don_model, 
            ae_model, 
            X_func_test_tensor[:n_samples], 
            X_loc_tensor, 
            pf_true_test, 
            device
        )
        
        print('--------Complete--------')

if __name__ == "__main__":
    main()