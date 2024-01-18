# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "预测"
__Created__ = 2023/12/22 11:37
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import InputFeatures, Dataset
from model import ColClassifierModel, CondClassifierModel
from utils import get_cond_op_dict, read_predict_datas, get_conn_op_dict, get_columns, get_agg_dict, get_values_name


def predict(columns, questions, predict_result_path, pretrain_model_path, column_model_path, value_model_path,
            hidden_size, batch_size, question_length, max_length, table_name='table_name'):
    # 创建模型
    col_model = ColClassifierModel(pretrain_model_path, hidden_size, len(get_agg_dict()), len(get_conn_op_dict()))
    cond_model = CondClassifierModel(pretrain_model_path, hidden_size, question_length)
    # 提取特征数据（不含label的数据）
    input_features = InputFeatures(pretrain_model_path, question_length, max_length).list_features(columns, questions)
    dataset = Dataset(input_features)
    # 预测不用打乱顺序shuffle=False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 是否使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        col_model = col_model.to(device)
        cond_model = cond_model.to(device)
        col_model.load_state_dict(torch.load(column_model_path, map_location=torch.device(device)))
        cond_model.load_state_dict(torch.load(value_model_path, map_location=torch.device(device)))
    # 预测
    pre_all_agg = []
    pre_all_conn_op = []
    pre_all_cond_cols = []
    pre_all_cond_ops = []
    pre_all_cond_vals = []
    pre_all_cond_counts = []
    for input_ids, attention_mask, token_type_ids, cls_idx in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        out_agg, out_conn_op = col_model(input_ids, attention_mask, token_type_ids, cls_idx)
        # 取预测结果最大值，torch.argmax找到指定纬度最大值所对应的索引（是索引，不是值）
        pre_agg = torch.argmax(out_agg, dim=2).cpu().numpy()
        pre_conn_op = torch.argmax(out_conn_op, dim=1).cpu().numpy()
        pre_all_agg.extend(pre_agg)
        pre_all_conn_op.extend(pre_conn_op)
    for input_ids, attention_mask, token_type_ids, cls_idx in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        out_cond_cols, out_cond_ops, out_cond_vals, out_cond_count = cond_model(input_ids, attention_mask,
                                                                                token_type_ids)
        pre_cond_cols = torch.argmax(out_cond_cols, dim=2).cpu().numpy()
        pre_cond_ops = torch.argmax(out_cond_ops, dim=2).cpu().numpy()
        pre_cond_vals = torch.argmax(out_cond_vals, dim=2).cpu().numpy()
        pre_cond_count = torch.argmax(out_cond_count, dim=1).cpu().numpy()
        pre_all_cond_cols.extend(pre_cond_cols)
        pre_all_cond_ops.extend(pre_cond_ops)
        pre_all_cond_vals.extend(pre_cond_vals)
        pre_all_cond_counts.extend(pre_cond_count)

    with open(predict_result_path, 'w', encoding='utf-8') as wf:
        for question, agg, conn_op, cond_cols, cond_ops, cond_vals, cond_counts in zip(questions, pre_all_agg,
                                                                                       pre_all_conn_op,
                                                                                       pre_all_cond_cols,
                                                                                       pre_all_cond_ops,
                                                                                       pre_all_cond_vals,
                                                                                       pre_all_cond_counts):
            sel_col = np.where(np.array(agg) != get_agg_dict()['none'])[0]
            agg = agg[agg != get_agg_dict()['none']]
            cond_col = cond_cols[cond_cols <= len(columns)]
            cond_op = cond_ops[cond_ops != get_cond_op_dict()['none']]
            sel_col_name = [columns[idx_col] for idx_col in sel_col]
            cond_vals_name = get_values_name(question[0], cond_vals)
            conds = [[int(cond_col), int(cond_op), cond_vals_name] for
                     cond_col, cond_op, cond_vals_name in zip(range(cond_counts), cond_col, cond_op, cond_vals_name)]
            sql_dict = {"question": question, "table_id": table_name,
                        "sql": {"sel": list(map(int, sel_col)),
                                "agg": list(map(int, agg)),
                                "limit": 0,
                                "orderby": [],
                                "asc_desc": 0,
                                "cond_conn_op": int(conn_op),
                                'conds': conds},
                        "keywords": {"sel_cols": sel_col_name, "values": cond_vals_name}}
            sql_json = json.dumps(sql_dict, ensure_ascii=False)
            wf.write(sql_json + '\n')


if __name__ == '__main__':
    hidden_size = 768
    batch_size = 12
    question_length = 128
    max_length = 512
    table_path = '../train-datas/table.xlsx'
    predict_question_path = '../train-datas/train_test.jsonl'
    predict_result_path = '../result-predict/predict.jsonl'
    pretrain_model_path = '../bert-base-chinese-hgd'
    column_model_path = '../result-model/classifier-column-model.pkl'
    value_model_path = '../result-model/classifier-value-model.pkl'
    columns = get_columns(table_path)
    questions = read_predict_datas(predict_question_path)
    predict(columns, questions, predict_result_path, pretrain_model_path, column_model_path, value_model_path,
            hidden_size, batch_size, question_length, max_length)
