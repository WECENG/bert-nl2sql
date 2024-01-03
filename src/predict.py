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
from model import ColClassifierModel, ValueClassifierModel
from utils import get_cond_op_dict, read_predict_datas, get_conn_op_dict, get_columns, get_agg_dict, get_values_by_idx


def predict(questions, predict_result_path, pretrain_model_path, column_model_path, value_model_path, hidden_size,
            batch_size, question_length, max_length, table_name='table_name'):
    # 创建模型
    col_model = ColClassifierModel(pretrain_model_path, hidden_size, len(get_agg_dict()), len(get_conn_op_dict()),
                                   len(get_cond_op_dict()))
    value_model = ValueClassifierModel(pretrain_model_path, hidden_size, question_length)
    # 提取特征数据（不含label的数据）
    input_features = InputFeatures(pretrain_model_path, question_length, max_length).list_features(questions)
    dataset = Dataset(input_features)
    # 是否使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        col_model = col_model.to(device)
        value_model = value_model.to(device)
        col_model.load_state_dict(torch.load(column_model_path, map_location=torch.device(device)))
        value_model.load_state_dict(torch.load(value_model_path, map_location=torch.device(device)))
    # 预测不用打乱顺序shuffle=False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 预测
    pre_all_agg = []
    pre_all_conn_op = []
    pre_all_cond_ops = []
    value_pre_cond_ops = []
    pre_all_cond_vals = []
    for input_ids, attention_mask, token_type_ids, cls_idx in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        out_agg, out_cond_ops, out_conn_op = col_model(input_ids, attention_mask, token_type_ids, cls_idx)

        # 取预测结果最大值，torch.argmax找到指定纬度最大值所对应的索引（是索引，不是值）
        pre_agg = torch.argmax(out_agg, dim=2).cpu().numpy()
        pre_cond_ops = torch.argmax(out_cond_ops, dim=2).cpu().numpy()
        pre_conn_op = torch.argmax(out_conn_op, dim=1).cpu().numpy()

        pre_all_agg.extend(pre_agg)
        pre_all_cond_ops.extend(pre_cond_ops)
        pre_all_conn_op.extend(pre_conn_op)
        value_pre_cond_ops.append(pre_cond_ops)

    for data, cond_ops in tqdm(zip(dataloader, value_pre_cond_ops), total=len(dataloader)):
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        token_type_ids = data[2].to(device)
        out_cond_vals = value_model(input_ids, attention_mask, token_type_ids, cond_ops, device)

        pre_all_cond_vals.extend(out_cond_vals)

    with open(predict_result_path, 'w', encoding='utf-8') as wf:
        for question, agg, conn_op, cond_ops, cond_vals in zip(questions, pre_all_agg, pre_all_conn_op,
                                                               pre_all_cond_ops, pre_all_cond_vals):
            sel_col = np.where(np.array(agg) != get_agg_dict()['none'])[0]
            agg = agg[agg != get_agg_dict()['none']]
            cond_col = np.where(np.array(cond_ops) != get_cond_op_dict()['none'])[0]
            cond_op = cond_ops[cond_ops != get_cond_op_dict()['none']]
            sel_col_name = [get_columns()[idx_col] for idx_col in sel_col]
            cond_vals_name = [result
                              for value_idx_start, value_idx_end in cond_vals
                              if (result := get_values_by_idx(question, value_idx_start.item(),
                                                              value_idx_end.item(),
                                                              conn_op)) is not None]
            conds = [[int(item_cond_col), int(item_cond_op), item_cond_val_name] for
                     item_cond_col, item_cond_op, item_cond_val_name in zip(cond_col, cond_op, cond_vals_name)]
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
    batch_size = 24
    question_length = 128
    max_length = 512
    predict_question_path = '../train-datas/waic_nl2sql_testa_public.jsonl'
    predict_result_path = '../result-predict/predict.jsonl'
    pretrain_model_path = '../bert-base-chinese-hgd'
    column_model_path = '../result-model/classifier-column-model.pkl'
    value_model_path = '../result-model/classifier-value-model.pkl'
    questions = read_predict_datas(predict_question_path)
    predict(questions, predict_result_path, pretrain_model_path, column_model_path, value_model_path, hidden_size,
            batch_size, question_length, max_length)
