import os.path
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
    task = 'reliable'
    dataset = json.load(open(f'../datasets/{dataset_name}.json'))

    indices = list(range(len(dataset)))
    cnt = 0
    misleading_cnt = 0
    total = 0

    for index in indices:
        item = dataset[index]

        candidate = []
        # for debate_index in range(5):
        #     try:
        #         conflict = json.load(open(f'res/mediQ_easy_gpt-4o/{index}/conlict_{debate_index}.json'))
        #         judgment = json.load(open(f'res/mediQ_easy_gpt-4o/{index}/judgment_{debate_index}.json'))
        #     except:
        #         continue
        #     candidate.append(conflict['source_idx_{}'.format(judgment['incorrect_idx'])])

        misleading = json.load(open(f'res/{dataset_name}_{model}/{index}/misleading.json'))
        if not os.path.exists(f'res/{dataset_name}_{model}/{index}/{task}_context_decision.json'):
            continue
        answer = json.load(open(f'res/{dataset_name}_{model}/{index}/{task}_context_decision.json'))
        misleading_idx = misleading['misleading_idx']
        if misleading_idx is not None:
            misleading_idx = int(misleading_idx)
        else:
            misleading_idx = 0
        if item['misleading_idx'] == misleading_idx:
            misleading_cnt += 1
        try:
            if item['golden_answer_idx'] == answer['choice']:
                cnt += 1
        except:
            continue
        total += 1

    print(cnt / total)
    print(misleading_cnt / total)


if __name__ == '__main__':
    main()
