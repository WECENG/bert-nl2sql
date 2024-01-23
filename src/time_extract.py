import re
from datetime import datetime, timedelta
from functools import reduce

import jieba.posseg as psg
from dateutil.relativedelta import relativedelta


def time_extract(text):
    replace_text = text
    time_res = []
    word = ''
    key_date = {'前天': -2, '前日': -2, '昨天': -2, '昨日': -2, '今天': 0, '今日': 0, '明天': 1, '明日': 1, '后天': 2,
                '后日': 2}
    key_month = {'上月': -1, '上个月': -1, '这月': 0, '本月': 0, '下月': 1, '下个月': 1}
    key_year = {'去年': -1, '今年': 0, '本年': 0, '明年': 1}

    for k, v in psg.cut(text):
        if k in key_date:
            if word != '':
                time_res.append(word)
            offset_date = (datetime.today() + timedelta(days=key_date.get(k, 0)))
            replace_text = text.replace(k, offset_date.strftime('%Y-%m-%d'))
            word = offset_date.strftime('%Y年%m月%d日')
        elif k in key_month:
            if word != '':
                time_res.append(word)
            offset_date = (datetime.today() + relativedelta(months=key_month.get(k, 0)))
            replace_text = text.replace(k, offset_date.strftime('%Y-%m'))
            word = offset_date.strftime('%Y年%m月')
        elif k in key_year:
            if word != '':
                time_res.append(word)
            offset_date = (datetime.today() + relativedelta(years=key_year.get(k, 0)))
            replace_text = text.replace(k, offset_date.strftime('%Y'))
            word = offset_date.strftime('%Y年')
        elif word != '':
            if v in ['m', 't']:
                word = word + k
            else:
                time_res.append(word)
                word = ''
        elif v in ['m', 't']:
            word = k
    if word != '':
        time_res.append(word)
    result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
    final_res = [parse_datetime(w) for w in result]
    replace_text = reduce(lambda text, replacement: text.replace(*replacement) if replacement[1] is not None else text,
                          zip(time_res, final_res), replace_text)
    return replace_text, [x for x in final_res if x is not None]


def check_time_valid(word):
    m = re.match("\d+$", word)
    if m:
        if len(word) <= 6:
            return None
    word1 = re.sub('[号|日]\d+$', '日', word)
    if word1 != word:
        return check_time_valid(word1)
    else:
        return word1


def parse_datetime(msg):
    if msg is None or len(msg) == 0:
        return None

    m = re.match(
        r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([上中下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二三四五六七八九十百]+分?)?([0-9零一二三四五六七八九十百]+秒)?",
        msg)
    if m.group(0) is not None and m.group(0) is not '':
        res = {
            "year": m.group(1),
            "month": m.group(2),
            "day": m.group(3),
            "hour": m.group(5) if m.group(5) is not None else '00',
            "minute": m.group(6) if m.group(6) is not None else '00',
            "second": m.group(7) if m.group(7) is not None else '00',
        }
        params = {}

        for name in res:
            if res[name] is not None and len(res[name]) != 0:
                if name == 'year':
                    tmp = year2dig(res[name][:-1])
                else:
                    tmp = cn2dig(res[name][:-1])
                if tmp is not None:
                    params[name] = int(tmp)
        target_date = datetime.today().replace(**params)
        is_pm = m.group(4)
        if is_pm is not None:
            if is_pm == u'下午' or is_pm == u'晚上' or is_pm == '中午':
                hour = target_date.time().hour
                if hour < 12:
                    target_date = target_date.replace(hour=hour + 12)
        formatted_result = None
        if res['day'] is None and res['month'] is None:
            formatted_result = target_date.strftime('%Y')
        elif res['day'] is None and res['month'] is not None:
            formatted_result = target_date.strftime('%Y-%m')
        elif res['day'] is not None and res['hour'] == '00' and res['minute'] == '00' and res['second'] == '00':
            formatted_result = target_date.strftime('%Y-%m-%d')
        elif res['day'] is not None:
            formatted_result = target_date.strftime('%Y-%m-%d %H:%M:%S')
        return formatted_result
    else:
        return None


UTIL_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}
UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def cn2dig(src):
    if src == "":
        return None
    m = re.match("\d+", src)
    if m:
        return int(m.group(0))
    rsl = 0
    unit = 1
    for item in src[::-1]:
        if item in UTIL_CN_UNIT.keys():
            unit = UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM.keys():
            num = UTIL_CN_NUM[item]
            rsl += num * unit
        else:
            return None
    if rsl < unit:
        rsl += unit
    return rsl


def year2dig(year):
    res = ''
    for item in year:
        if item in UTIL_CN_NUM.keys():
            res = res + str(UTIL_CN_NUM[item])
        else:
            res = res + item
    m = re.match("\d+", res)
    if m:
        if len(m.group(0)) == 2:
            return int(datetime.today().year / 100) * 100 + int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None


if __name__ == '__main__':
    print(time_extract('上个月的收益'))

    print(time_extract('今天的收益'))

    print(time_extract('去年的收益'))

    text1 = '我要住到明天下午三点'
    print(text1, time_extract(text1), sep=':')

    text2 = '预定28号的房间'
    print(text2, time_extract(text2), sep=':')

    text3 = '我要从26号下午4点住到11月2号'
    print(text3, time_extract(text3), sep=':')
