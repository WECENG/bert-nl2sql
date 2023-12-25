# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "预测"
__Created__ = 2023/12/22 11:37
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import InputFeatures, Dataset
from model import ColClassifierModel, ValueClassifierModel
from utils import get_cond_op_dict, read_predict_datas, get_conn_op_dict


def predict(questions, pretrain_model_path, column_model_path, value_model_path, hidden_size, batch_size,
            question_length, max_length):
    # 创建模型
    col_model = ColClassifierModel(pretrain_model_path, hidden_size, len(get_cond_op_dict()))
    value_model = ValueClassifierModel(pretrain_model_path, hidden_size, len(get_conn_op_dict()), question_length)
    # 提取特征数据（不含label的数据）
    input_features = InputFeatures(pretrain_model_path, question_length, max_length).list_features(questions)
    dataset = Dataset(input_features)
    # 是否使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        col_model = col_model.to(device)
        col_model.load_state_dict(torch.load(column_model_path, map_location=torch.device(device)))
        value_model.load_state_dict(torch.load(column_model_path, map_location=torch.device(device)))
    # 预测不用打乱顺序shuffle=False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 预测
    pre_all_sel_col = []
    pre_all_cond_col = []
    pre_all_cond_op = []
    pre_all_conn_op = []
    pre_all_cond_values = []
    for input_ids, attention_mask, token_type_ids, cls_idx in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        out_sel_col, out_cond_col, out_cond_op = col_model(input_ids, attention_mask, token_type_ids, cls_idx)

        # 取预测结果最大值，torch.argmax找到指定纬度最大值所对应的索引（是索引，不是值）
        pre_sel_col = torch.argmax(out_sel_col.data, dim=1).cpu().numpy()
        pre_cond_col = torch.argmax(out_cond_col.data, dim=1).cpu().numpy()
        pre_cond_op = torch.argmax(out_cond_op.data, dim=1).cpu().numpy()

        pre_all_sel_col.append(pre_sel_col)
        pre_all_cond_col.append(pre_cond_col)
        pre_all_cond_op.append(pre_cond_op)

    for input_ids, attention_mask, token_type_ids, _ in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        out_conn_op, out_cond_values = value_model(input_ids, attention_mask, token_type_ids)

        # 取预测结果最大值，torch.argmax找到指定维度最大值所对应的索引（是索引，不是值）
        pre_conn_op = torch.argmax(out_conn_op.data, dim=1).cpu().numpy()
        pre_cond_values = torch.argmax(out_cond_values.data, dim=1).cpu().numpy()

        pre_all_conn_op.append(pre_conn_op)
        pre_all_cond_values.append(pre_cond_values)

    print("pre_all_sel_col data:", pre_all_sel_col)
    print("pre_all_cond_col data:", pre_all_cond_col)
    print("pre_all_cond_op data:", pre_all_cond_op)
    print("pre_all_cond_op data:", pre_all_conn_op)
    print("pre_all_cond_op data:", pre_all_cond_values)


if __name__ == '__main__':
    hidden_size = 768
    batch_size = 24
    question_length = 128
    max_length = 512
    predict_question_path = '../train-datas/waic_nl2sql_testa_public.jsonl'
    pretrain_model_path = '../bert-base-chinese-hgd'
    column_model_path = '../result-model/classifier-column-model.pkl'
    value_model_path = '../result-model/classifier-value-model.pkl'
    questions = read_predict_datas(predict_question_path)
    predict(questions, pretrain_model_path, column_model_path, value_model_path, hidden_size, batch_size,
            question_length, max_length)
