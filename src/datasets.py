# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "数据集"
__Created__ = 2023/12/14 14:52
"""
from typing import List

import numpy as np
import torch.utils.data
from transformers import BertTokenizer

from utils import get_columns


# label
class Label(object):
    def __init__(self, label_agg: List = None, label_conn_op=None, label_cond_ops: List = None,
                 label_cond_vals: List = None):
        """
        训练标签信息
        :param label_agg: 聚合函数
        :param label_conn_op: 连接操作符
        :param label_cond_ops: 条件操作符
        :param label_cond_vals: 条件值
        """
        self.label_agg = label_agg
        self.label_conn_op = label_conn_op
        self.label_cond_ops = label_cond_ops
        self.label_cond_vals = label_cond_vals


class InputFeatures(object):
    def __init__(self, model_path=None, question_length=128, max_length=512, input_ids=None, attention_mask=None,
                 token_type_ids=None, cls_idx=None, label: Label = None):
        if model_path is not None:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.question_length = question_length
        self.max_length = max_length
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_idx = cls_idx
        self.label = label

    def encode_columns(self, columns: List):
        """
        列编码
        :param columns: 列
        :return: 编码后的列，及序列号（用于列与列之间的区分）
        """
        columns_encode = []
        segment_ids = []
        i = 1
        for column in columns:
            encod = self.tokenizer.encode(column)
            seg = [i] * len(encod)
            columns_encode.extend(encod)
            segment_ids.extend(seg)
            i = 1 - i  # 切换 0 和 1
        return torch.tensor(columns_encode), torch.tensor(segment_ids)

    def get_cls_idx(self, columns):
        """
        获取列标记符的位置
        :param columns: 列
        :return:
        """
        cls_idx = []
        start = self.question_length
        for i in range(len(columns)):
            cls_idx.append(int(start))
            # 加上特殊标记的长度（例如 [CLS] 和 [SEP]）
            start += len(columns[i]) + 2
        return cls_idx

    def encode_question_with_columns(self, que_length, max_length, question, columns_encode, columns_segment_id):
        """
        编码
        :param que_length: 问题长度
        :param max_length: text长度
        :param question:  问题
        :param columns_encode:  编码的列
        :param columns_segment_id 编码的列的序列
        :return: 编码后的text
        """

        # 编码问题，需要填充，否则会出现长度不一致异常
        question_encoding = self.tokenizer.encode(question, add_special_tokens=True, padding='max_length',
                                                  max_length=que_length, truncation=True)

        # 合并编码后的张量，保证张量类型(dtype)为int或long, bert的embedding的要求
        input_ids = torch.cat([torch.tensor(question_encoding), columns_encode], dim=0)
        token_type_ids = torch.cat([torch.zeros(que_length, dtype=torch.long), columns_segment_id], dim=0)
        padding_length = max_length - len(input_ids)
        attention_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(padding_length)], dim=0)
        input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)], dim=0)
        token_type_ids = torch.cat([token_type_ids, torch.zeros(padding_length, dtype=torch.long)], dim=0)

        return input_ids, attention_mask, token_type_ids

    def list_features(self, datas):
        """
        输入特征
        :param datas: 数据
        :param que_length: 问题长度
        :param max_length: text长度
        :return: 特征信息
        """
        list_features = []
        columns = get_columns()
        cls_idx = self.get_cls_idx(columns)
        columns_encode, columns_segment_id = self.encode_columns(get_columns())
        for data in datas:
            # if contain label data
            label = None
            if len(data) > 1:
                label = Label(label_agg=data[1], label_conn_op=data[2], label_cond_ops=data[3], label_cond_vals=data[4])
            question = data[0]
            # 编码(question+columns)
            input_ids, attention_mask, token_type_ids = self.encode_question_with_columns(self.question_length,
                                                                                          self.max_length,
                                                                                          question, columns_encode,
                                                                                          columns_segment_id)
            list_features.append(
                InputFeatures(question_length=self.question_length, max_length=self.max_length, input_ids=input_ids,
                              attention_mask=attention_mask, token_type_ids=token_type_ids, cls_idx=cls_idx,
                              label=label))
        return list_features


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features: List[InputFeatures]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        cls_idx = np.array(feature.cls_idx)
        if feature.label is not None:
            label: Label = feature.label
            label_agg = np.array(label.label_agg)
            label_conn_op = np.array(label.label_conn_op)
            label_cond_ops = np.array(label.label_cond_ops)
            label_cond_vals = np.array([np.array(val) for val in label.label_cond_vals])
            return input_ids, attention_mask, token_type_ids, cls_idx, label_agg, label_conn_op, label_cond_ops, label_cond_vals
        else:
            return input_ids, attention_mask, token_type_ids, cls_idx
