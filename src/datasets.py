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

from utils import get_columns, get_cond_op_dict


# sql查询条件
class Conditions(object):
    def __init__(self, cond_col=None, cond_op=None, cond_value=None):
        """
        example [cond_col cond_op cond_value]
        :param cond_col: sql的查询条件列
        :param cond_op: sql的查询条件操作符
        :param cond_value: sql的查询条件值
        """
        self.cond_col = cond_col
        self.cond_op = cond_op
        self.cond_value = cond_value


# label
class Label(object):
    def __init__(self, label_sel_col=None, label_conn_op=None, label_cond: List[Conditions] = None):
        """
        example [select label_sel_col from table where label_condition[0] label_conn_op label_condition[1] label_conn_op ...]
        :param label_sel_col:
        :param label_conn_op:
        :param label_cond:
        """
        self.label_sel_col = label_sel_col
        self.label_conn_op = label_conn_op
        self.label_cond = label_cond


class InputFeatures(object):
    def __init__(self, model_path=None, question_length=128, max_length=512, input_ids=None, attention_mask=None,
                 token_type_ids=None, cls_idx=None, label=None):
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
                label = Label(label_sel_col=[data[1]], label_conn_op=[data[3]],
                              label_cond=[Conditions(cond[0], cond[1], cond[2:4]) for cond in data[2]] if data[
                                                                                                              2] is not None else None)
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
            label_sel_col = np.array(label.label_sel_col)
            label_conn_op = np.array(label.label_conn_op)
            label_cond = np.array(label.label_cond)
            if label_cond.any() is None or label_cond.size == 0:
                # 初始化一维数组，保证纬度一致
                # 52对应‘无’这一列
                label_cond_col = np.array([len(get_columns()) - 1], dtype=np.int32)
                # 7对应‘none’操作符
                label_cond_op = np.array([len(get_cond_op_dict()) - 1], dtype=np.int32)
                label_cond_value = np.array([[0, 0]], dtype=np.int32)
            else:
                # 转化成一维数组，保证纬度一致
                label_cond_col = np.array([[item.cond_col] for item in label_cond]).ravel()[:np.prod(1)].reshape(
                    1)
                label_cond_op = np.array([[item.cond_op] for item in label_cond]).ravel()[:np.prod(1)].reshape(
                    1)
                label_cond_value = np.array([item.cond_value for item in label_cond]).ravel()[:np.prod((1, 2))].reshape(
                    (1, 2))
            # 打印样本信息
            print(f"Sample {item}:")
            print(f"input_ids shape: {input_ids.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"token_type_ids shape: {token_type_ids.shape}")
            print(f"label_sel_col shape: {label_sel_col.shape}")
            print(f"label_conn_op shape: {label_conn_op.shape}")
            print(f"label_cond_col shape: {label_cond_col.shape}")
            print(f"label_cond_op shape: {label_cond_op.shape}")
            print(f"label_cond_value shape: {label_cond_value.shape}")
            return input_ids, attention_mask, token_type_ids, cls_idx, label_sel_col, label_conn_op, label_cond_col, label_cond_op, label_cond_value
        else:
            return input_ids, attention_mask, token_type_ids, cls_idx
