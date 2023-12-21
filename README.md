# 自然语言转SQL

将NLP语言转为机器可以理解的SQL语言

模型的输入为一个 Question，输出一个 SQL 结构，该 SQL 结构对应一条 SQL 语句。

SQL结构

- `sel` 为一个 list，代表 `SELECT` 语句所选取的列
- `agg` 为一个 list，与 `sel` 一一对应，表示对该列做哪个聚合操作，比如 sum, max, min 等
- `conds` 为一个 list，代表 `WHERE` 语句中的的一系列条件，每个条件是一个由 (条件列，条件运算符，条件值) 构成的三元组
- `cond_conn_op` 为一个 int，代表 `conds` 中各条件之间的并列关系，可以是 and 或者 or

```json
{
	"id": 830061772,
	"question": "诊断下持有酒鬼酒的基金",
	"table_id": "FundTable",
	"sql": {
		"sel": [1],  # 查询列，对应基金表的列，如本例中代表的是第一列（以0开始）
		"agg": [0],  # 查询列对应的聚合类型
		"limit": 0,
		"orderby": [],
		"asc_desc": 0, 
		"cond_conn_op": 0,
		"conds": [
			[51, 4, "%酒鬼酒%"]  # [基金表的条件列, 条件操作符, 条件值]
		]
	},
	"keywords": {
		"sel_cols": ["基金名称"],
		"values": ["%酒鬼酒%"]
	}
}

操作符和关键词字典如下：
op = {'>': 0, '<': 1, '==': 2, '!=': 3, 'like': 4, '>=': 5, '<=': 6, 'none': 7}
agg = {'none': 0, 'avg': 1, 'max': 2, 'min': 3, 'count': 4, 'sum': 5}
connect_op = {'none': 0, 'and': 1, 'or': 2}
```

