# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "工具包"
__Created__ = 2023/12/18 16:37
"""
import json


def read_train_datas(path, question_length):
    """
    :param path 数据路径
    :param question_length 问题长度
    :return: [[question, agg, conn_op, cond_ops, cond_vals],...], cond_vals:[[val_start_idx,val_end_idx],...]
    """
    column_length = len(get_columns())
    with open(path, 'r', encoding='utf-8') as f:
        data_list = []
        for line in f:
            item = json.loads(line)
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
            # cond_ops & cond_vals
            cond_ops = [get_cond_op_dict()['none']] * column_length
            cond_vals = [0] * question_length
            if item['sql'].get('conds') is not None:
                conds = item['sql']['conds']
                for i, cond in enumerate(conds):
                    cond_col_item = cond[0]
                    cond_op_item = cond[1]
                    cond_ops[cond_col_item] = cond_op_item
                    value = cond[2]
                    cond_vals = fill_value_start_end(cond_vals, question, value)
            data_list.append([question, agg, conn_op, cond_ops, cond_vals])
    return data_list


def read_predict_datas(path):
    """
    :param path: 预测数据路径
    :return: 预测数据
    """
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question = item['question']
            questions.append([question])
    return questions


def fill_value_start_end(cond_vals, question, value):
    """
    fill [1] by the value in the question
    """
    question_length = len(question)
    value_length = len(value)
    for i in range(question_length - value_length + 1):
        if question[i:value_length + i] == value:
            cond_vals[i: value_length + i] = [1] * value_length
    return cond_vals


def get_columns():
    columns = ['基金代码', '基金名称', '成立时间', '基金类型', '基金规模', '销售状态', '是否可销售', '风险等级',
               '基金公司名称', '分红方式',
               '赎回状态', '是否支持定投', '净值同步日期', '净值', '成立以来涨跌幅', '昨日涨跌幅', '近一周涨跌幅',
               '近一个月涨跌幅', '近三个月涨跌幅', '近六个月涨跌幅',
               '近一年涨跌幅', '基金经理', '主题/概念', '一个月夏普率', '一年夏普率', '三个月夏普率', '六个月夏普率',
               '成立以来夏普率', '投资市场', '板块', '行业',
               '晨星三年评级', '管理费率', '销售服务费率', '托管费率', '认购费率', '申购费率', '赎回费率', '分红年度',
               '权益登记日',
               '除息日', '派息日', '红利再投日', '每十份收益单位派息', '主投资产类型', '基金投资风格描述', '估值',
               '是否主动管理型基金', '投资', '跟踪指数',
               '是否新发', '重仓', '无']
    return columns


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


def count_values(cond_vals):
    """
   cond_vals的值如[0,1,1,1,1,0,0,0,1,1,1,0,0,0]所示
   统计出现1的数量，续的1只统计一次
   """
    count = 0
    pre = 0
    for idx, val in enumerate(cond_vals):
        if val == 1 and pre == 0:
            count = count + 1
            pre = 1
        else:
            pre = 0
    return count


def get_values_name(question, cond_vals):
    """
    cond_vals的值如[0,1,1,1,1,0,0,0,1,1,1,0,0,0]所示
    根据cond_vals中为1的值找到question对应下标的内容
    返回找到的内容列表，连续为1的内容作为返回列表的一个元素
    """
    question = question
    result = []
    cur_start_idx = 0
    valid = False

    for idx, val in enumerate(cond_vals):
        if val == 1:
            if not valid:
                cur_start_idx = idx
                valid = True
        else:
            if valid:
                valid = False
                if idx > cur_start_idx:
                    vals = question[cur_start_idx:idx]
                    result.append(vals)

    return result
