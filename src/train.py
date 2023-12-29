# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "训练"
__Created__ = 2023/12/14 16:25
"""
import numpy as np
import torch.cuda
from sklearn import metrics
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import Dataset, InputFeatures
from model import ColClassifierModel, ValueClassifierModel
from utils import read_train_datas, get_conn_op_dict, get_cond_op_dict, get_agg_dict


def train(model: ColClassifierModel or ValueClassifierModel, model_save_path, train_dataset: Dataset,
          val_dataset: Dataset, batch_size, lr, epochs):
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # 是否使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)
    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)
    best_val_avg_acc = 0
    for epoch in range(epochs):
        total_loss_train = 0
        model.train()
        # 训练进度
        for input_ids, attention_mask, token_type_ids, cls_idx, label_agg, label_conn_op, label_cond_ops, label_cond_vals in tqdm(
                train_loader):
            # model要求输入的矩阵(hidden_size,sequence_size),需要把第二纬度去除.squeeze(1)
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.squeeze(1).to(device)
            if type(model) is ColClassifierModel:
                label_agg = label_agg.to(dtype=torch.float, device=device)
                label_conn_op = label_conn_op.to(dtype=torch.float, device=device)
                label_cond_ops = label_cond_ops.to(dtype=torch.float, device=device)
                # 模型输出
                out_agg, out_cond_ops, out_conn_op = model(input_ids, attention_mask, token_type_ids, cls_idx)
                out_agg = out_agg.argmax(dim=2).to(dtype=torch.float, device=device)
                out_cond_ops = out_cond_ops.argmax(dim=2).to(dtype=torch.float, device=device)
                out_conn_op = out_conn_op.argmax(dim=1).to(dtype=torch.float, device=device)
                # 计算损失
                loss_agg = criterion(label_agg, out_agg)
                loss_conn_op = criterion(out_conn_op, label_conn_op)
                loss_cond_ops = criterion(out_cond_ops, label_cond_ops)
                # todo 损失比例
                total_loss_train = loss_agg + loss_conn_op + loss_cond_ops
                total_loss_train.requires_grad_(True)
            if type(model) is ValueClassifierModel:
                label_cond_vals = label_cond_vals.to(dtype=torch.float, device=device)
                # cond_ops中不为'none'操作的数量
                label_cond_counts = np.count_nonzero(label_cond_ops != get_cond_op_dict()['none'], axis=1)
                # 模型输出
                out_cond_vals = model(input_ids, attention_mask, token_type_ids)
                # 按照cond_ops中对应的数量n提取出out_cond_vals前n个最大值
                cond_vals = [torch.topk(out_cond_vals[i], k=label_cond_counts[i], dim=0, largest=True).indices for i in
                             range(len(out_cond_vals))]
                # 按照label_cond_vals中不为[0,0]的元素位置进行填充
                cond_vals_filled = []
                for label_cond_val, cond_val in zip(label_cond_vals, cond_vals):
                    cond_idx = 0
                    cond_val_fill = torch.zeros((len(label_cond_val), 2), dtype=torch.int)
                    # 使用 enumerate 获取索引和值
                    for i, cond_item in enumerate(label_cond_val):
                        # detach()为了不影响梯度计算
                        if not torch.equal(cond_item.clone().detach(), torch.zeros((2,), dtype=torch.float).to(device)):
                            cond_val_fill[i] = cond_val[cond_idx]
                            cond_idx += 1
                    cond_vals_filled.append(cond_val_fill)
                out_cond_vals = torch.stack(cond_vals_filled, dim=0).to(dtype=torch.float,
                                                                        device=device)
                # 计算损失
                lost_cond_vals = criterion(out_cond_vals, label_cond_vals)
                total_loss_train = lost_cond_vals
                total_loss_train.requires_grad_(True)
            # 模型更新
            model.zero_grad()
            optim.zero_grad()
            total_loss_train.backward()
            optim.step()
        # 模型验证
        val_avg_acc = 0
        out_all_agg = []
        out_all_conn_op = []
        out_all_cond_ops = []
        out_all_cond_vals = []
        label_all_agg = []
        label_all_conn_op = []
        label_all_cond_ops = []
        label_all_cond_vals = []
        # 验证无需梯度计算
        model.eval()
        with torch.no_grad():
            # 使用当前epoch训练好的模型验证
            for input_ids, attention_mask, token_type_ids, cls_idx, label_agg, label_conn_op, label_cond_ops, label_cond_vals in val_loader:
                input_ids = input_ids.squeeze(1).to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.squeeze(1).to(device)
                if type(model) is ColClassifierModel:
                    label_agg = label_agg.to(dtype=torch.float, device=device)
                    label_conn_op = label_conn_op.to(dtype=torch.float, device=device)
                    label_cond_ops = label_cond_ops.to(dtype=torch.float, device=device)
                    # 模型输出
                    # 模型输出
                    out_agg, out_cond_ops, out_conn_op = model(input_ids, attention_mask, token_type_ids, cls_idx)
                    out_agg = out_agg.argmax(dim=2).to(device)
                    out_cond_ops = out_cond_ops.argmax(dim=2).to(device)
                    out_conn_op = out_conn_op.argmax(dim=1).to(device)
                    out_all_agg.append(out_agg.cpu().numpy())
                    out_all_conn_op.append(out_conn_op.cpu().numpy())
                    out_all_cond_ops.append(out_cond_ops.cpu().numpy())
                    label_all_agg.append(label_agg.cpu().numpy())
                    label_all_conn_op.append(label_conn_op.cpu().numpy())
                    label_all_cond_ops.append(label_cond_ops.cpu().numpy())
                if type(model) is ValueClassifierModel:
                    # reshape(-1)需要转成一维数组才能计算准确率
                    label_cond_vals = label_cond_vals.squeeze(1).to(dtype=torch.float, device=device).reshape(-1)
                    # 模型输出
                    out_cond_vals = model(input_ids, attention_mask, token_type_ids)
                    out_all_cond_vals.append(out_cond_vals.argmax(dim=1).reshape(-1).cpu().numpy())
                    label_all_cond_vals.append(label_cond_vals.cpu().numpy())

        if type(model) is ColClassifierModel:
            val_agg_acc = metrics.accuracy_score(np.concatenate(np.concatenate(out_all_agg)),
                                                 np.concatenate(np.concatenate(label_all_agg)))
            val_conn_op_acc = metrics.accuracy_score(np.concatenate(out_all_conn_op),
                                                     np.concatenate(label_all_conn_op))
            val_cond_ops_acc = metrics.accuracy_score(np.concatenate(np.concatenate(out_all_cond_ops)),
                                                      np.concatenate(np.concatenate(label_all_cond_ops)))
            # todo 准确率计算逻辑
            val_avg_acc = (val_agg_acc + val_conn_op_acc + val_cond_ops_acc) / 3
        if type(model) is ValueClassifierModel:
            val_cond_vals_acc = metrics.accuracy_score(np.concatenate(out_all_cond_vals),
                                                       np.concatenate(label_all_cond_vals))
            val_avg_acc = val_cond_vals_acc
        # save model
        if val_avg_acc > best_val_avg_acc:
            best_val_avg_acc = val_avg_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'''best model | Val Accuracy: {best_val_avg_acc: .3f}''')
        print(
            f'''Epochs: {epoch + 1} 
              | Train Loss: {total_loss_train: .3f} ]
              | Val Accuracy: {val_avg_acc: .3f}''')


def test(model, model_save_path, test_dataset, batch_size):
    # 加载最佳模型权重
    model.load_state_dict(torch.load(model_save_path))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model = model.to(device)

    total_acc_test = 0
    model.eval()
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            attention_mask = test_input['attention_mask'].to(device)
            input_ids = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_ids, attention_mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_dataset): .3f}')


if __name__ == '__main__':
    hidden_size = 768
    batch_size = 24
    learn_rate = 1e-5
    epochs = 5
    question_length = 128
    max_length = 512
    train_data_path = '../train-datas/waic_nl2sql_train.jsonl'
    pretrain_model_path = '../bert-base-chinese'
    save_column_model_path = '../result-model/classifier-column-model.pkl'
    save_value_model_path = '../result-model/classifier-value-model.pkl'
    # 加载数据
    label_datas = read_train_datas(train_data_path)
    # 提取特征数据
    list_input_features = InputFeatures(pretrain_model_path, question_length, max_length).list_features(label_datas)
    # 初始化dataset
    dateset = Dataset(list_input_features)
    # 创建模型
    col_model = ColClassifierModel(pretrain_model_path, hidden_size, len(get_agg_dict()), len(get_conn_op_dict()),
                                   len(get_cond_op_dict()))
    value_model = ValueClassifierModel(pretrain_model_path, hidden_size, question_length)
    # 分割数据集
    total_size = len(label_datas)
    train_size = int(0.001 * total_size)
    val_size = int(0.001 * total_size)
    test_size = total_size - train_size - val_size
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(dateset, [train_size, val_size, test_size])
    print('train column model begin')
    train(col_model, save_column_model_path, train_dataset, val_dataset, batch_size, learn_rate, epochs)
    print('train column model finish')
    print('train value model begin')
    train(value_model, save_value_model_path, train_dataset, val_dataset, batch_size, learn_rate,
          epochs)
    print('train value model finish')
