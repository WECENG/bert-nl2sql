# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "模型"
__Created__ = 2023/12/14 14:50
"""
import numpy as np
import torch
from torch import nn
from transformers import BertModel

from utils import get_cond_op_dict


class ColClassifierModel(nn.Module):
    def __init__(self, model_path, hidden_size, agg_length, conn_op_length, cond_op_length, dropout=0.5):
        super(ColClassifierModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        # out classes需要纬度必须大于label中size(classes)，否则会出现Assertion `t >= 0 && t < n_classes` failed.
        self.agg_classifier = nn.Linear(hidden_size, agg_length)
        self.cond_ops_classifier = nn.Linear(hidden_size, cond_op_length)
        self.conn_op_classifier = nn.Linear(hidden_size, conn_op_length)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, cls_idx=None):
        # 输出最后一层隐藏状态以及池化层
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        dropout_output = self.dropout(outputs.pooler_output)
        dropout_hidden_state = self.dropout(outputs.last_hidden_state)

        """
        提取列特征信息，从dim=1即第二维中（列标记符号索引所在纬度）提取dropout_hidden_state对应该纬度的信息。
        前提需要将cls_idx张量shape扩展成与dropout_hidden_state一致
        """
        # cls_cols = dropout_hidden_state.gather(dim=1, index=cls_idx.unsqueeze(-1).expand(
        #     dropout_hidden_state.shape[0], -1, dropout_hidden_state.shape[-1]))
        # 简化写法
        cls_cols = dropout_hidden_state[:, cls_idx[0], :]

        out_agg = self.agg_classifier(cls_cols)
        out_cond_ops = self.cond_ops_classifier(cls_cols)

        out_conn_op = self.conn_op_classifier(dropout_output)

        return out_agg, out_cond_ops, out_conn_op


class ValueClassifierModel(nn.Module):
    def __init__(self, model_path, hidden_size, question_length, cond_value_length=2, dropout=0.5):
        super(ValueClassifierModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.cond_vals_classifier = nn.Linear(hidden_size, cond_value_length)
        self.question_length = question_length

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, cond_ops=None,
                device='cpu'):
        # 输出最后一层隐藏状态以及池化层
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        dropout_hidden_state = self.dropout(outputs.last_hidden_state)

        # 提取问题特征信息
        cond_values = dropout_hidden_state[:, 1:int(self.question_length), :]

        out_cond_vals = self.cond_vals_classifier(cond_values)

        # cond_ops中不为'none'操作的数量
        cond_counts = np.count_nonzero(cond_ops != get_cond_op_dict()['none'], axis=1)

        # 按照cond_ops中对应的数量n提取出out_cond_vals前n个最大值
        valid_cond_vals = [torch.topk(out_cond_vals[i], k=cond_counts[i], dim=0, largest=True).indices for i in
                           range(len(out_cond_vals))]
        # 按照label_cond_vals中不为[0,0]的元素位置进行填充
        cond_vals_filled = []
        for cond_op, valid_cond_val in zip(cond_ops, valid_cond_vals):
            cond_idx = 0
            cond_val_fill = torch.zeros((len(cond_op), 2), dtype=torch.float, device=device)
            # 使用 enumerate 获取索引和值
            for i, cond_item in enumerate(cond_op):
                if not cond_item == get_cond_op_dict()['none']:
                    cond_val_fill[i] = valid_cond_val[cond_idx]
                    cond_idx += 1
            cond_vals_filled.append(cond_val_fill)
        out_cond_vals = torch.stack(cond_vals_filled, dim=0).to(dtype=torch.float)

        return out_cond_vals
