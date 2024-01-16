import json


if __name__ == '__main__':
    with open('../train-datas/train.jsonl', 'r', encoding='utf-8') as f:
        with open('../train-datas/train.jsonl', 'w', encoding='utf-8') as nf:
            for idx, line in enumerate(f):
                item = json.loads(line)
                item['id'] = idx + 1
                nf.write(json.dumps(item, ensure_ascii=False) + "\n")

