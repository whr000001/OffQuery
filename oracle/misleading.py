import os
import json
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils import obtain_response, obtain_json, extract_dict
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='mediQ_easy')
parser.add_argument('--model', type=str, default='gpt-4o')
args = parser.parse_args()


dataset_name = args.dataset
model = args.model


def find_misleading(each_information):
    with open('misleading.txt') as f:
        prompt = f.read()
    inputs = prompt

    agent_text = ''
    for agent_index, agent_information in enumerate(each_information):
        agent_text += 'Agent {}: {}\n'.format(agent_index, agent_information)

    inputs = inputs.replace('<===agent===>', agent_text)
    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    if not isinstance(output, dict):
        output = extract_dict(output, ['misleading_idx', 'reason'])
    return output


def main():
    data = json.load(open(f'../datasets/{dataset_name}.json'))

    indices = list(range(len(data)))

    out = []
    save_path = f'res/{dataset_name}_{model}_misleading.json'
    if os.path.exists(save_path):
        out = json.load(open(save_path))

    for index in tqdm(indices[len(out):]):
        item = data[index]
        misleading = find_misleading(item['shuffled_private_information'])
        out.append(misleading)
        json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()
