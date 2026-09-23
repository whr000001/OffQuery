import re
import json


def main():
    dataset_name = 'mediQ_easy'
    model = 'gpt-5-mini'
    dataset = json.load(open(f'../datasets/{dataset_name}.json'))
    answers = json.load(open(f'res/{dataset_name}_{model}_decision.json'))
    misleading = json.load(open(f'res/{dataset_name}_{model}_misleading.json'))

    indices = list(range(len(dataset)))
    cnt = 0
    misleading_cnt = 0

    total = 0

    for index in indices:
        item = dataset[index]
        try:
            if item['misleading_idx'] == int(misleading[index]['misleading_idx']):
                misleading_cnt += 1
        except:
            continue
        if item['golden_answer_idx'] == answers[index]['choice']:
            cnt += 1
        total += 1

    print(cnt / total)
    print(misleading_cnt / total)


if __name__ == '__main__':
    main()
