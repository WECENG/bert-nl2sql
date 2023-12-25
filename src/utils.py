# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "工具包"
__Created__ = 2023/12/18 16:37
"""
import json


def read_train_datas(path):
    """
    :return: [[question, sel_col, conds:[col, op, start, end], conn_op],...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data_list = []
        for line in f:
            item = json.loads(line)
            question = item['question']
            sel = item['sql']['sel'][0]
            cond_conn_op = item['sql']['cond_conn_op']
            if item['sql'].get('conds') is not None:
                conds = item['sql']['conds']
                for i, cond in enumerate(conds):
                    value = cond[2]
                    start, end = value_start_end(question, value)
                    cond[2] = start
                    cond.append(end)
            else:
                conds = None
            data_list.append([question, sel, conds, cond_conn_op])
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


def value_start_end(question, value):
    """
    get the start and end index of the value in the question
    """
    question_length = len(question)
    value_length = len(value)
    for i in range(question_length - value_length + 1):
        if question[i:value_length + i] == value:
            return i, i + value_length - 1
    return 0, 0


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
