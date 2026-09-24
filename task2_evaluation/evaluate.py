import os
import json
from tqdm import tqdm
from utils import obtain_response, obtain_json
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='mediQ_hard')
parser.add_argument('--task', type=str, default='unreliable')
parser.add_argument('--model', type=str, default='qwen3-32b')
parser.add_argument('--baseline', type=str, default='ours')
args = parser.parse_args()


dataset_name = args.dataset
task = args.task
assert task in ['unreliable', 'reliable']
model = args.model
baseline = args.baseline


def evaluate(text, facts):
    with open('prompts/evaluate.txt') as f:
        prompt = f.read()
    facts_text = ''
    for index, fact in enumerate(facts):
        facts_text += 'Fact {}: {}\n'.format(index, fact)
    inputs = prompt
    inputs = inputs.replace('<===text===>', text)
    inputs = inputs.replace('<===facts===>', facts_text)
    output = obtain_response(inputs, model='gpt-4o')
    output = obtain_json(output)
    return output


def main():
    golden = json.load(open(f'truth/{dataset_name}.json'))
    data = json.load(open(f'../datasets/{dataset_name}.json'))
    indices = list(range(len(golden)))

    if baseline != 'oracle':
        save_path = f'evaluations/{dataset_name}_{model}_{baseline}_{task}.json'
    else:
        save_path = f'evaluations/{dataset_name}_{model}_{baseline}.json'
    if os.path.exists(save_path):
        out = json.load(open(save_path))
    else:
        out = []
    for index in tqdm(indices[len(out):]):
        if baseline == 'ours':
            if not os.path.exists(f'../ours/res/{dataset_name}_{model}/{index}/{task}_context_regeneration_max_3.json'):
                out.append(None)
                continue
            context = json.load(open(f'../ours/res/{dataset_name}_{model}/{index}/{task}_context_regeneration_max_3.json'))
        elif baseline == 'vanilla':
            if not os.path.exists(f'../vanilla/res/{dataset_name}_{model}/{index}/{task}_context_corrected.json'):
                out.append(None)
                continue
            context = json.load(open(f'../vanilla/res/{dataset_name}_{model}/{index}/{task}_context_corrected.json'))
        elif baseline == 'oracle':
            context = data[index]['original_context']
        else:
            raise KeyError
        try:
            out.append(evaluate(context, golden[index]['sampled_facts']))
        except:
            out.append(None)
        json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()
