# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "模型"
__Created__ = 2023/12/14 14:50
"""
from torch import nn
from transformers import BertModel


class ColClassifierModel(nn.Module):
    def __init__(self, model_path, hidden_size, cond_op_length, dropout=0.5):
        super(ColClassifierModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        # todo 可以不止一列
        self.sel_col_classifier = nn.Linear(hidden_size, 1)
        # todo 条件不止一列
        self.cond_col_classifier = nn.Linear(hidden_size, 1)
        # out classes需要纬度必须大于label中size(classes)，否则会出现Assertion `t >= 0 && t < n_classes` failed.
        self.cond_op_classifier = nn.Linear(hidden_size, cond_op_length)
        self.relu = nn.ReLU()

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

        out_sel_col = self.sel_col_classifier(cls_cols).squeeze(-1)
        out_sel_col = self.relu(out_sel_col)
        out_cond_col = self.cond_col_classifier(cls_cols).squeeze(-1)
        out_cond_col = self.relu(out_cond_col)

        out_cond_op = self.cond_op_classifier(dropout_output)

        return out_sel_col, out_cond_col, out_cond_op


class ValueClassifierModel(nn.Module):
    def __init__(self, model_path, hidden_size, max_value_length, conn_op_length, dropout=0.):
        super(ValueClassifierModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        # todo 最大条件值数量
        self.cond_values_classifier = nn.Linear(hidden_size, max_value_length)
        self.conn_op_classifier = nn.Linear(hidden_size, conn_op_length)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # 输出最后一层隐藏状态以及池化层
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        dropout_output = self.dropout(outputs.pooler_output)
        dropout_hidden_state = self.dropout(outputs.last_hidden_state)

        out_conn_op = self.conn_op_classifier(dropout_output)

        # 提取条件列值特征信息
        cond_values = dropout_hidden_state[:, 1:64, :]

        out_cond_values = self.cond_values_classifier(cond_values)

        return out_conn_op, out_cond_values
