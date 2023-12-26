# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "预测"
__Created__ = 2023/12/22 11:37
"""
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import InputFeatures, Dataset
from model import ColClassifierModel, ValueClassifierModel
from utils import get_cond_op_dict, read_predict_datas, get_conn_op_dict, get_values_by_idx, get_columns


def get_topk_values(out_cond_value, k):
    """
    获取每行的前k个最大值
    :param out_cond_value: tensor值
    :param k:获取的每行最大值的数量
    :return: 每行的前k个最大值
    """
    topk_indices = torch.topk(out_cond_value, k, dim=1, largest=True).indices
    pre_cond_values = topk_indices.cpu().numpy().tolist()
    return pre_cond_values


def predict(questions, predict_result_path, pretrain_model_path, column_model_path, value_model_path, hidden_size,
            batch_size, question_length, max_length, k=2, table_name='table_name'):
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
        value_model = value_model.to(device)
        col_model.load_state_dict(torch.load(column_model_path, map_location=torch.device(device)))
        value_model.load_state_dict(torch.load(value_model_path, map_location=torch.device(device)))
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

        pre_all_sel_col.extend(pre_sel_col)
        pre_all_cond_col.extend(pre_cond_col)
        pre_all_cond_op.extend(pre_cond_op)

    for input_ids, attention_mask, token_type_ids, _ in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        out_conn_op, out_cond_values = value_model(input_ids, attention_mask, token_type_ids)

        # 取预测结果最大值，torch.argmax找到指定维度最大值所对应的索引（是索引，不是值）
        pre_conn_op = torch.argmax(out_conn_op.data, dim=1).cpu().numpy()
        pre_cond_values = get_topk_values(out_cond_values, k)

        pre_all_conn_op.extend(pre_conn_op)
        pre_all_cond_values.extend(pre_cond_values)

    with open(predict_result_path, 'w', encoding='utf-8') as wf:
        for i in range(len(questions)):
            question = questions[i][0]
            sel_col = pre_all_sel_col[i].item()
            sel_col_name = get_columns()[sel_col]
            cond_col = pre_all_cond_col[i].item()
            cond_op = pre_all_cond_op[i].item()
            cond_values = pre_all_cond_values[i]
            conn_op = pre_all_conn_op[i].item()
            real_value1, real_value2 = get_values_by_idx(question, cond_values[0], cond_values[1], conn_op)
            if real_value1 == '' and real_value2 == '':
                dict_str = {"question": question, "table_id": table_name,
                            "sql": {"sel": [sel_col], "agg": [0], "limit": 0, "orderby": [], "asc_desc": 0,
                                    "cond_conn_op": 0}, "keywords": {"sel_cols": [sel_col_name], "values": []}}
                json_str = json.dumps(dict_str, ensure_ascii=False)
                wf.write(json_str + '\n')
            elif real_value1 != '':
                if real_value2 != '':
                    cond = [[cond_col, cond_op, real_value1], [cond_col, cond_op, real_value2]]
                    values = [real_value1, real_value2]
                else:
                    cond = [[cond_col, cond_op, real_value1]]
                    values = [real_value1]
                    conn_op = 0
                dict_str = {"question": question, "table_id": table_name,
                            "sql": {"sel": [sel_col], "agg": [0], "limit": 0, "orderby": [], "asc_desc": 0,
                                    "cond_conn_op": conn_op, 'conds': cond},
                            "keywords": {"sel_cols": [sel_col_name], "values": values}}
                json_str = json.dumps(dict_str, ensure_ascii=False)
                wf.write(json_str + '\n')

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
    predict_result_path = '../result-predict/predict.jsonl'
    pretrain_model_path = '../bert-base-chinese-hgd'
    column_model_path = '../result-model/classifier-column-model.pkl'
    value_model_path = '../result-model/classifier-value-model.pkl'
    questions = read_predict_datas(predict_question_path)
    predict(questions, predict_result_path, pretrain_model_path, column_model_path, value_model_path, hidden_size,
            batch_size, question_length, max_length)
