import json
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    dataset_name = 'mediQ_easy'
    model = 'qwen3-32b'
    baseline = 'ours'
    task = 'unreliable'

    golden = json.load(open(f'truth/{dataset_name}.json'))
    indices = list(range(len(golden)))

    if baseline == 'oracle':
        answer_path = f'evaluations/{dataset_name}_{model}_{baseline}.json'
    else:
        answer_path = f'evaluations/{dataset_name}_{model}_{baseline}_{task}.json'
    answers = json.load(open(answer_path))

    all_acc = []
    for index in indices:
        prediction = [0] * len(golden[index]['golden_truth'])
        if answers[index] is None:
            all_acc.append(0)
            continue
        for item in answers[index]:
            if item < len(prediction):
                prediction[item] = 1
        all_acc.append(accuracy_score(golden[index]['golden_truth'], prediction))
    print(np.mean(all_acc))


if __name__ == '__main__':
    main()
