import re
import os
import json
from utils import obtain_response, obtain_json, extract_dict
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='mediQ_hard')
parser.add_argument('--model', type=str, default='gpt-4o')
parser.add_argument('--task', type=str, default='unreliable')
args = parser.parse_args()


dataset_name = args.dataset
model = args.model
task = args.task
assert task in ['unreliable', 'reliable']


def independent(information):
    with open('prompts/independent.txt') as f:
        prompt = f.read()
    inputs = prompt
    inputs = inputs.replace('<===context===>', information)
    output = obtain_response(inputs, model=model)
    return output


def discuss(previous, others):
    with open('prompts/discuss.txt') as f:
        prompt = f.read()
    others_text = ''
    for item in others:
        others_text += '{}\n'.format(item)
    inputs = prompt
    inputs = inputs.replace('<===previous===>', previous)
    inputs = inputs.replace('<===others===>', others_text)
    output = obtain_response(inputs, model=model)
    return output


def find_misleading(each_information):
    with open('prompts/misleading.txt') as f:
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


def correct(shared_information, statements):
    with open('prompts/correct.txt') as f:
        prompt = f.read()
    inputs = prompt
    statement_text = ''
    for item in statements:
        statement_text += '{}\n'.format(item)
    inputs = inputs.replace('<===context===>', shared_information)
    inputs = inputs.replace('<===options===>', statement_text)
    output = obtain_response(inputs, model=model)
    return output


def decision_making(context, question, options):
    with open(f'prompts/decision_making.txt', 'r') as f:
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


def run(shared_information, agent_information, question, options, save_dir):
    agent_indices = list(range(len(agent_information)))

    agent_indices = list(range(len(agent_information)))

    if os.path.exists(f'{save_dir}/all_statements.json'):
        all_statements = json.load(open(f'{save_dir}/all_statements.json'))
    else:
        all_statements = []
        for agent_index in agent_indices:
            statements = independent(agent_information[agent_index])
            all_statements.append(statements)
        json.dump(all_statements, open(f'{save_dir}/all_statements.json', 'w'))

    discussion_rounds = 3
    for discussion_index in range(discussion_rounds):
        if os.path.exists(f'{save_dir}/discussion_round_{discussion_index}.json'):
            all_statements = json.load(open(f'{save_dir}/discussion_round_{discussion_index}.json'))
        else:
            new_statements = []
            for i in range(len(all_statements)):
                others = []
                for j in range(len(all_statements)):
                    if i == j:
                        continue
                    others.append(all_statements[j])
                new_statements.append(discuss(all_statements[i], others))
            all_statements = new_statements
            json.dump(all_statements, open(f'{save_dir}/discussion_round_{discussion_index}.json', 'w'))

    if os.path.exists(f'{save_dir}/misleading.json'):
        misleading = json.load(open(f'{save_dir}/misleading.json'))
    else:
        misleading = find_misleading(all_statements)
        json.dump(misleading, open(f'{save_dir}/misleading.json', 'w'))

    misleading_idx = misleading['misleading_idx']
    if misleading_idx is not None:
        misleading_idx = int(misleading_idx)
    else:
        misleading_idx = 0
    trusted_statements = []
    for index, statement in enumerate(all_statements):
        if index != misleading_idx:
            trusted_statements.append(statement)

    if os.path.exists(f'{save_dir}/{task}_context_corrected.json'):
        context = json.load(open(f'{save_dir}/{task}_context_corrected.json'))
    else:
        context = correct(shared_information, trusted_statements)
        json.dump(context, open(f'{save_dir}/{task}_context_corrected.json', 'w'))

    if os.path.exists(f'{save_dir}/{task}_context_decision.json'):
        return
    else:
        decision = decision_making(context, question, options)
        json.dump(decision, open(f'{save_dir}/{task}_context_decision.json', 'w'))


def run_one(index, item):
    save_dir = f'res/{dataset_name}_{model}/{index}'

    os.makedirs(save_dir, exist_ok=True)

    run(
        item['unreliable_context'] if task == 'unreliable' else item['shared_information'],
        item['shuffled_private_information'],
        item['question'],
        item['options'],
        save_dir
    )
    return index


def main():
    data = json.load(open(f'../datasets/{dataset_name}.json'))

    indices = list(range(len(data)))

    if not os.path.exists(f'res/{dataset_name}_{model}'):
        os.makedirs(f'res/{dataset_name}_{model}')

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_one, index, data[index]): index
            for index in indices
        }
        # tqdm 显示已经完成的 run 数量
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                index = futures[future]

                try:
                    future.result()
                except Exception as e:
                    print(f'\nError in index {index}: {e}')

                pbar.update(1)


if __name__ == '__main__':
    main()
