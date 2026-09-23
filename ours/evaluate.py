import re
import json


def obtain(original):
    if isinstance(original, dict):
        return original['choice'], int(original['misleading agent'])
    else:
        choice, misleading_agent = None, None
        for i in original:
            if i in ['A', 'B', 'C', 'D'] and choice is None:
                choice = i
            if i.isdigit() and misleading_agent is None:
                misleading_agent = int(i)
        if choice is None:
            choice = 'A'
        if misleading_agent is None:
            misleading_agent = 0
        return choice, misleading_agent


def obtain_index(item):
    if isinstance(item, int):
        return item
    if item is None:
        print('---')
        return 0
    item = re.search(r'[0-9]', item)
    if item is None:
        print('---')
        return 0
    return int(item.group())


def main():
    dataset_name = 'mediQ_hard'
    model = 'gpt-4o'
    dataset = json.load(open(f'../datasets/{dataset_name}.json'))
    task = 'unreliable'
    indices = list(range(len(dataset)))
    cnt = 0
    misleading_cnt = 0
    total = 0

    size = 1
    for index in indices:
        item = dataset[index]

        candidate = []

        try:
            misleading = json.load(open(f'res/{dataset_name}_{model}/{index}/misleading_max_{size}.json'))
            answer = json.load(open(f'res/{dataset_name}_{model}/{index}/{task}_context_decision_max_{size}.json'))
        except:
            continue
        # if item['misleading_idx'] == int(misleading['statement_idx']):
        #     misleading_cnt += 1
        if item['golden_answer_idx'] == answer['choice']:
            cnt += 1
        total += 1

    print((cnt) / total)
    print(misleading_cnt / total)
    print(total)


if __name__ == '__main__':
    main()
