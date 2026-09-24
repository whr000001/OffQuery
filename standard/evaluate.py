import os.path
import re
import json


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
        misleading = json.load(open(f'res/{dataset_name}_{model}/{index}/misleading.json'))
        answer = json.load(open(f'res/{dataset_name}_{model}/{index}/{task}_context_decision.json'))
        misleading_idx = misleading['misleading_idx']
        misleading_idx = int(misleading_idx)
        if item['misleading_idx'] == misleading_idx:
            misleading_cnt += 1
        if item['golden_answer_idx'] == answer['choice']:
            cnt += 1
        total += 1

    print(cnt / total)
    print(misleading_cnt / total)


if __name__ == '__main__':
    main()
