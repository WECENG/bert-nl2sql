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
from utils import read_train_datas, get_conn_op_dict, get_cond_op_dict


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
        for input_ids, attention_mask, token_type_ids, cls_idx, label_sel_col, label_conn_op, label_cond_col, label_cond_op, label_cond_value in tqdm(
                train_loader):
            # model要求输入的矩阵(hidden_size,sequence_size),需要把第二纬度去除.squeeze(1)
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.squeeze(1).to(device)
            if type(model) is ColClassifierModel:
                label_sel_col = label_sel_col.squeeze(-1).to(device)
                label_cond_col = label_cond_col.squeeze(-1).to(device)
                label_cond_op = label_cond_op.squeeze(-1).to(device)
                # 模型输出
                out_sel_col, out_cond_col, out_cond_op = model(input_ids, attention_mask, token_type_ids, cls_idx)
                # 计算损失
                loss_sel_col = criterion(out_sel_col, label_sel_col)
                loss_cond_col = criterion(out_cond_col, label_cond_col)
                loss_cond_op = criterion(out_cond_op, label_cond_op)
                # todo 损失比例
                total_loss_train = loss_sel_col + loss_cond_col + loss_cond_op
            if type(model) is ValueClassifierModel:
                label_conn_op = label_conn_op.squeeze(-1).to(device)
                label_cond_values = label_cond_value.squeeze(1).to(device)
                # 模型输出
                out_conn_op, out_cond_values = model(input_ids, attention_mask, token_type_ids)
                # 计算损失
                lost_conn_op = criterion(out_conn_op, label_conn_op)
                lost_cond_values = criterion(out_cond_values, label_cond_values)
                # todo 损失比例
                total_loss_train = lost_conn_op + lost_cond_values
            # 模型更新
            model.zero_grad()
            optim.zero_grad()
            total_loss_train.backward()
            optim.step()
        # 模型验证
        val_avg_acc = 0
        out_all_sel_col = []
        out_all_cond_col = []
        out_all_cond_op = []
        label_all_sel_col = []
        label_all_cond_col = []
        label_all_cond_op = []
        out_all_conn_op = []
        out_all_cond_values = []
        label_all_conn_op = []
        label_all_cond_values = []
        # 验证无需梯度计算
        model.eval()
        with torch.no_grad():
            # 使用当前epoch训练好的模型验证
            for input_ids, attention_mask, token_type_ids, cls_idx, label_sel_col, label_conn_op, label_cond_col, label_cond_op, label_cond_value in val_loader:
                input_ids = input_ids.squeeze(1).to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.squeeze(1).to(device)
                if type(model) is ColClassifierModel:
                    label_sel_col = label_sel_col.squeeze(-1).to(device)
                    label_cond_col = label_cond_col.squeeze(-1).to(device)
                    label_cond_op = label_cond_op.squeeze(-1).to(device)
                    # 模型输出
                    out_sel_col, out_cond_col, out_cond_op = model(input_ids, attention_mask, token_type_ids, cls_idx)
                    out_all_sel_col.append(out_sel_col.argmax(dim=1).cpu().numpy())
                    out_all_cond_col.append(out_cond_col.argmax(dim=1).cpu().numpy())
                    out_all_cond_op.append(out_cond_op.argmax(dim=1).cpu().numpy())
                    label_all_sel_col.append(label_sel_col.cpu().numpy())
                    label_all_cond_col.append(label_cond_col.cpu().numpy())
                    label_all_cond_op.append(label_cond_op.cpu().numpy())
                if type(model) is ValueClassifierModel:
                    label_conn_op = label_conn_op.squeeze(-1).to(device)
                    # reshape(-1)需要转成一维数组才能计算准确率
                    label_cond_values = label_cond_value.squeeze(1).to(device).reshape(-1)
                    # 模型输出
                    out_conn_op, out_cond_values = model(input_ids, attention_mask, token_type_ids)
                    out_all_conn_op.append(out_conn_op.argmax(dim=1).cpu().numpy())
                    out_all_cond_values.append(out_cond_values.argmax(dim=1).reshape(-1).cpu().numpy())
                    label_all_conn_op.append(label_conn_op.cpu().numpy())
                    label_all_cond_values.append(label_cond_values.cpu().numpy())

        if type(model) is ColClassifierModel:
            val_sel_col_acc = metrics.accuracy_score(np.concatenate(out_all_sel_col), np.concatenate(label_all_sel_col))
            val_cond_col_acc = metrics.accuracy_score(np.concatenate(out_all_cond_col),
                                                      np.concatenate(label_all_cond_col))
            val_cond_op_acc = metrics.accuracy_score(np.concatenate(out_all_cond_op), np.concatenate(label_all_cond_op))
            # todo 准确率计算逻辑
            val_avg_acc = (val_sel_col_acc + val_cond_col_acc + val_cond_op_acc) / 3
        if type(model) is ValueClassifierModel:
            val_conn_op_acc = metrics.accuracy_score(np.concatenate(out_all_conn_op), np.concatenate(label_all_conn_op))
            val_cond_values_acc = metrics.accuracy_score(np.concatenate(out_all_cond_values),
                                                         np.concatenate(label_all_cond_values))
            val_avg_acc = (val_conn_op_acc + val_cond_values_acc) / 2
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
    # 加载数据
    label_datas = read_train_datas('../train-datas/waic_nl2sql_train.jsonl')
    # 提取特征数据
    list_input_features = InputFeatures('../bert-base-chinese', question_length, max_length).list_features(label_datas)
    # 初始化dataset
    dateset = Dataset(list_input_features)
    # 创建模型
    colModel = ColClassifierModel('../bert-base-chinese', hidden_size, len(get_cond_op_dict()))
    valueModel = ValueClassifierModel('../bert-base-chinese', hidden_size, 2, len(get_conn_op_dict()), question_length)
    # 分割数据集
    total_size = len(label_datas)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(dateset, [train_size, val_size, test_size])
    # print('train column model begin')
    # train(colModel, '../result-model/classifier-model.pkl', train_dataset, val_dataset, batch_size, learn_rate, epochs)
    # print('train column model finish')
    print('train value model begin')
    train(valueModel, '../result-model/classifier-model.pkl', train_dataset, val_dataset, batch_size, learn_rate,
          epochs)
    print('train value model finish')
