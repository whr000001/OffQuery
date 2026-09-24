import re
import json


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

        misleading = json.load(open(f'res/{dataset_name}_{model}/{index}/misleading_max_{size}.json'))
        answer = json.load(open(f'res/{dataset_name}_{model}/{index}/{task}_context_decision_max_{size}.json'))
  
        if item['golden_answer_idx'] == answer['choice']:
            cnt += 1
        total += 1

    print((cnt) / total)
    print(misleading_cnt / total)
    print(total)


if __name__ == '__main__':
    main()
