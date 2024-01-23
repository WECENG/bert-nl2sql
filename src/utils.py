# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "工具包"
__Created__ = 2023/12/18 16:37
"""
import json
import pandas as pd
import nltk
import jieba.posseg as psg

from nltk.corpus import stopwords
from time_extract import time_extract


def read_train_datas(path, question_length, columns):
    """
    :param path 数据路径
    :param question_length 问题长度
    :param columns 列
    :return: [[question, agg, conn_op, cond_ops, cond_vals],...], cond_vals:[[val_start_idx,val_end_idx],...]
    """
    column_length = len(columns)
    with open(path, 'r', encoding='utf-8') as f:
        data_list = []
        for line in f:
            item = json.loads(line)
            time_replace(item)
            # question
            question = item['question']
            # agg
            sel = item['sql']['sel']
            agg_op = item['sql']['agg']
            agg = [get_agg_dict()['none']] * column_length
            for i in range(len(sel)):
                sel_col_item = sel[i]
                agg_op_item = agg_op[i]
                agg[sel_col_item] = agg_op_item
            # conn_op
            conn_op = item['sql']['cond_conn_op']
            # cond_cols & cond_ops & cond_vals
            # +1 默认初始化为不存在的列, question_length需要大于column_length
            cond_cols = [column_length + 1] * question_length
            cond_ops = [get_cond_op_dict()['none']] * question_length
            cond_vals = [0] * question_length
            if item['sql'].get('conds') is not None:
                conds = item['sql']['conds']
                for idx, cond in enumerate(conds):
                    cond_cols[idx] = cond[0]
                    cond_ops[idx] = cond[1]
                    value = cond[2]
                    cond_vals = fill_value_start_end(cond_vals, question, value, idx)
            data_list.append([question, agg, conn_op, cond_cols, cond_ops, cond_vals])
    return data_list


def read_predict_datas(path):
    """
    :param path: 预测数据路径
    :return: 预测数据
    """
    origin_questions = []
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            origin_question = item['question']
            origin_questions.append([origin_question])
            time_replace(item)
            question = item['question']
            questions.append([question])
    return origin_questions, questions


def time_replace(data):
    """
    时间替换
    :param data: 数据
    :return: 数据
    """
    question, _ = time_extract(data['question'])
    data['question'] = question
    if 'keywords' in data and 'values' in data['keywords']:
        values = [time_extract(value)[0] for value in data['keywords']['values']]
        data['keywords']['values'] = values
    if 'sql' in data and 'conds' in data['sql']:
        conds = [[value[0], value[1], time_extract(value[2])[0]] for value in data['sql']['conds']]
        data['sql']['conds'] = conds


def stop_words():
    nltk.download('stopwords')
    words = stopwords.words('chinese')
    return words


def cut_words_first_end(cut_word_list, content):
    """
    剪切词
    :param cut_word_list: 剪切词列表
    :param content: 内容
    :return: 剪切后的内容
    """
    cut_result = [word for word, _ in psg.cut(content)]
    if len(cut_result) > 1 and cut_result[0] in cut_word_list:
        cut_result.pop(0)
    if len(cut_result) > 1 and cut_result[-1] in cut_word_list:
        cut_result.pop(-1)
    return ''.join(cut_result)


def get_columns(table_path):
    columns = pd.read_table(table_path, header=2)
    return columns.columns.__array__()


def get_cond_op_dict():
    cond_op_dict = {'>': 0, '<': 1, '==': 2, '!=': 3, 'like': 4, '>=': 5, '<=': 6, 'none': 7}
    return cond_op_dict


def get_conn_op_dict():
    conn_op_dict = {'none': 0, 'and': 1, 'or': 2}
    return conn_op_dict


def get_agg_dict():
    agg_dict = {'': 0, 'AVG': 1, 'MAX': 2, 'MIN': 3, 'COUNT': 4, 'SUM': 5, 'none': 6}
    return agg_dict


def get_key(dict, value):
    """
    根据字典的value获取key
    :param dict: 字典
    :param value: 值
    :return: key
    """
    return [k for k, v in dict.items() if v == value]


def fill_value_start_end(cond_vals, question, value, idx):
    """
    :param cond_vals 待填充待值
    :param question 问题
    :param value 待匹配待值
    :param 下标
    fill [1] by the value in the question
    结果类似[0,0,0,0,1,1,1,1,0,0,0,2,2,2,2,3,3,3,0,0,0,4,4,4,0,0,0,0,0]
    """
    question_length = len(question)
    value_length = len(value)
    for i in range(question_length - value_length + 1):
        if question[i:value_length + i] == value:
            cond_vals[i: value_length + i] = [idx + 1] * value_length
    return cond_vals


def count_values(cond_vals):
    """
   cond_vals的值如[0,0,0,0,1,1,1,1,0,0,0,2,2,2,2,3,3,3,0,0,0,4,4,4,0,0,0,0,0]所示
   统计出现>0的数量，连续的>0只统计一次
   """
    count = 0
    pre_value = None
    for idx, val in enumerate(cond_vals):
        if idx > 0:
            pre_value = cond_vals[idx - 1]
        if val > 0 and val != pre_value:
            count = count + 1
    return count


def get_values_name(question, cond_vals):
    """
    cond_vals的值如[0,0,0,0,1,1,1,1,0,0,0,2,2,2,2,3,3,3,0,0,0,4,4,4,0,0,0,0,0]所示
    根据cond_vals中为1的值找到question对应下标的内容
    返回找到的内容列表，连续为1的内容作为返回列表的一个元素
    """
    result = []
    start_of_segment = None
    for idx, current_value in enumerate(cond_vals):
        previous_value = cond_vals[idx - 1] if idx > 0 else None
        # 检测新的连续段的开始
        if current_value > 0 and current_value != previous_value and start_of_segment is None:
            start_of_segment = idx
        # 当段结束时，添加到结果并更新段的起始位置
        elif current_value != previous_value and start_of_segment is not None:
            segment = question[start_of_segment:idx]
            result.append(segment)
            start_of_segment = None if current_value == 0 else idx
    return result
