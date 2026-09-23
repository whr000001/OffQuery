import re
import os
import json
from utils import obtain_response, obtain_json, extract_dict
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='mediQ_easy')
parser.add_argument('--model', type=str, default='gpt-4o')
args = parser.parse_args()


dataset_name = args.dataset
model = args.model


def decision_making(context, question, options):
    with open(f'decision_making.txt', 'r') as f:
        prompt = f.read()
    options_text = ''
    for option_idx in ['A', 'B', 'C', 'D']:
        options_text += '{}. {}\n'.format(option_idx, options[option_idx])
    inputs = prompt
    inputs = inputs.replace('<===context===>', context)
    if question is not None:
        inputs = inputs.replace('<===question===>', question)
    inputs = inputs.replace('<===options===>', options_text)
    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    # print(inputs)
    if not isinstance(output, dict):
        output = extract_dict(output, ['choice', 'reason'])
    return output


def main():
    data = json.load(open(f'../datasets/{dataset_name}.json'))

    indices = list(range(len(data)))

    save_path = f'res/{dataset_name}_{model}_decision.json'
    if os.path.exists(save_path):
        out = json.load(open(save_path))
    else:
        out = []

    for index in tqdm(indices[len(out):]):
        item = data[index]
        out.append(decision_making(item['original_context'], question=item['question'], options=item['options']))
        json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()
