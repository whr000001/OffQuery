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
parser.add_argument('--max_debate_round', type=int, default=3)
args = parser.parse_args()


dataset_name = args.dataset
model = args.model
task = args.task
assert task in ['unreliable', 'reliable']

max_debate_round = args.max_debate_round


def extract_facts(private):
    with open('prompts/extract_facts.txt', encoding='utf-8') as f:
        prompt = f.read()
    inputs = prompt
    inputs = inputs.replace('<===context===>', private)
    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    if not isinstance(output, list):
        return []
    return output


def find_conflict(all_statements):
    with open('prompts/find_conflict.txt', encoding='utf-8') as f:
        prompt = f.read()
    text = ''
    for statements in all_statements:
        statement_text = ''
        for each in statements['statements']:
            statement_text += '  - {}\n'.format(each)
        text += 'Source {}\n{}'.format(statements['ID'], statement_text)

    inputs = prompt
    inputs = inputs.replace('<===statements===>', text)
    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    if "No contradiction" in output or '{' not in output:
        return output
    if not isinstance(output, dict):
        output = extract_dict(output, ['source_idx_0', 'source_idx_1', 'strength', 'reason'])
    return output


def debate_support(target_statement, statements):
    with open('prompts/debate_support.txt', encoding='utf-8') as f:
        prompt = f.read()
    statement_text = ''
    for statement_idx, statement in enumerate(statements):
        statement_text += 'Statement {}: {}\n'.format(statement_idx, statement)
    inputs = prompt
    inputs = inputs.replace('<===statements===>', statement_text)
    inputs = inputs.replace('<===statement===>', target_statement)
    output = obtain_response(inputs, model=model)
    return output


def debate_refute(target_statement, statements):
    with open('prompts/debate_refute.txt', encoding='utf-8') as f:
        prompt = f.read()
    statement_text = ''
    for statement_idx, statement in enumerate(statements):
        statement_text += 'Statement {}: {}\n'.format(statement_idx, statement)
    inputs = prompt
    inputs = inputs.replace('<===statements===>', statement_text)
    inputs = inputs.replace('<===statement===>', target_statement)
    output = obtain_response(inputs, model=model)
    return output


def judge(statement_0, statement_1, support_0, support_1, refute_0, refute_1):
    with open('prompts/judge.txt', encoding='utf-8') as f:
        prompt = f.read()
    inputs = prompt
    inputs = inputs.replace('<===statement_0===>', statement_0)
    inputs = inputs.replace('<===statement_1===>', statement_1)
    inputs = inputs.replace('<===support_0===>', support_0)
    inputs = inputs.replace('<===support_1===>', support_1)
    inputs = inputs.replace('<===refute_0===>', refute_0)
    inputs = inputs.replace('<===refute_1===>', refute_1)
    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    if not isinstance(output, dict):
        output = extract_dict(output, ['incorrect_idx', 'reason'])
    return output


def calculate_score(statement, facts):
    with open('prompts/calculate_score.txt', encoding='utf-8') as f:
        prompt = f.read()
    fact_text = ''
    for fact in facts:
        fact_text += '{}\n'.format(fact)
    inputs = prompt
    inputs = inputs.replace('<===facts===>', fact_text)
    inputs = inputs.replace('<===statement===>', statement)
    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    if not isinstance(output, dict):
        output = extract_dict(output, ['score', 'reason'])
    return output


def detect_misleading(facts, candidates, scores):
    with open('prompts/detect_misleading.txt', encoding='utf-8') as f:
        prompt = f.read()

    fact_text = ''
    for fact in facts:
        fact_text += '{}\n'.format(fact)

    statement_text = ''
    for item, score in zip(candidates, scores):
        each_text = ''
        for fact in item['statements']:
            each_text += '{} '.format(fact)
        each = 'Statement {}\nContent: {}\nScore: {}\nReason: {}\n'.format(item['ID'], each_text,
                                                                           score['score'], score['reason'])
        statement_text += '{}\n'.format(each)

    inputs = prompt
    inputs = inputs.replace('<===facts===>', fact_text)
    inputs = inputs.replace('<===statements===>', statement_text)
    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    if not isinstance(output, dict):
        output = extract_dict(output, ['statement_idx', 'reason'])
    return output


def evaluate_facts(statement, facts):
    with open('prompts/evaluate_facts.txt', encoding='utf-8') as f:
        prompt = f.read()
    inputs = prompt
    evidence_text = ''
    for evidence in facts:
        evidence_text += '{}\n'.format(evidence)
    inputs = inputs.replace('<===statement===>', statement)
    inputs = inputs.replace('<===evidence===>', evidence_text)

    output = obtain_response(inputs, model=model)
    output = obtain_json(output)
    if not isinstance(output, dict):
        output = extract_dict(output, ['verification', 'importance', 'reason'])
    return output


def context_regeneration(facts):
    with open('prompts/context_regeneration.txt') as f:
        prompt = f.read()
    inputs = prompt
    fact_text = ''
    for fact in facts:
        fact_text += '{}\n'.format(fact)
    inputs = inputs.replace('<===facts===>', fact_text)
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


def obtain_incorrect_idx_from_judgment(judgment):
    if 'incorrect_idx' in judgment:
        output = judgment['incorrect_idx']
        if isinstance(output, int):
            return output
        if isinstance(output, str):
            pos0 = output.find('0')
            pos1 = output.find('1')
            if pos0 == -1 and pos1 == -1:
                output = 0
            elif pos0 == -1:
                output = 1
            elif pos1 == -1:
                output = 0
            else:
                output = 0 if pos0 < pos1 else 1
        return output
    else:
        output = json.dumps(judgment)
        pos0 = output.find('0')
        pos1 = output.find('1')
        if pos0 == -1 and pos1 == -1:
            output = 0
        elif pos0 == -1:
            output = 1
        elif pos1 == -1:
            output = 0
        else:
            output = 0 if pos0 < pos1 else 1
        return output


def run(shared_information, agent_information, question, options, save_dir):
    agent_indices = list(range(len(agent_information)))

    if os.path.exists(f'{save_dir}/all_statements.json'):
        all_statements = json.load(open(f'{save_dir}/all_statements.json'))
    else:
        all_statements = []
        for agent_index in agent_indices:
            statements = extract_facts(agent_information[agent_index])
            all_statements.append(
                {
                    'ID': agent_index,
                    'statements': statements,
                }
            )
        json.dump(all_statements, open(f'{save_dir}/all_statements.json', 'w'))

    original_all_statements = all_statements

    for debate_index in range(max_debate_round):
        if os.path.exists(f'{save_dir}/conlict_{debate_index}.json'):
            conflict = json.load(open(f'{save_dir}/conlict_{debate_index}.json'))
        else:
            conflict = find_conflict(all_statements)
            json.dump(conflict, open(f'{save_dir}/conlict_{debate_index}.json', 'w'))
        # print(conflict)
        if isinstance(conflict, str) or 'source_idx_0' not in conflict or conflict['source_idx_0'] is None:
            break
        agent_0 = int(conflict['source_idx_0'])
        agent_1 = int(conflict['source_idx_1'])

        other_statements = []
        statement_0 = ''
        statement_1 = ''
        for fact in all_statements:
            if int(fact['ID']) == agent_0:
                for each in fact['statements']:
                    statement_0 += '{} '.format(each)
            elif int(fact['ID']) == agent_1:
                for each in fact['statements']:
                    statement_1 += '{} '.format(each)
            else:
                for each in fact['statements']:
                    other_statements.append(each)

        if os.path.exists(f'{save_dir}/debate_{debate_index}.json'):
            support_0, support_1, refute_0, refute_1 = json.load(open(f'{save_dir}/debate_{debate_index}.json'))
        else:
            support_0 = debate_support(statement_0, other_statements)
            support_1 = debate_support(statement_1, other_statements)
            refute_0 = debate_refute(statement_0, other_statements)
            refute_1 = debate_refute(statement_1, other_statements)
            json.dump([support_0, support_1, refute_0, refute_1], open(f'{save_dir}/debate_{debate_index}.json', 'w'))

        if os.path.exists(f'{save_dir}/judgment_{debate_index}.json'):
            judgment = json.load(open(f'{save_dir}/judgment_{debate_index}.json'))
        else:
            judgment = judge(statement_0, statement_1, support_0, support_1, refute_0, refute_1)
            json.dump(judgment, open(f'{save_dir}/judgment_{debate_index}.json', 'w'))
        removed_id = agent_0 if obtain_incorrect_idx_from_judgment(judgment) == 0 else agent_1
        new_all_statements = []
        for item in all_statements:
            if item['ID'] == removed_id:
                continue
            new_all_statements.append(item)
        all_statements = new_all_statements
    verified_facts = []
    for statements in all_statements:
        for statement in statements['statements']:
            verified_facts.append(statement)

    trustworthy_idx = [item['ID'] for item in all_statements]
    removed_candidate = []
    for item in original_all_statements:
        if item['ID'] not in trustworthy_idx:
            removed_candidate.append(item)

    if os.path.exists(f'{save_dir}/scores_max_{max_debate_round}.json'):
        scores = json.load(open(f'{save_dir}/scores_max_{max_debate_round}.json', 'r'))
    else:
        scores = []
        for statements in removed_candidate:
            statement_text = ''
            for statement in statements['statements']:
                statement_text += '{} '.format(statement)
            scores.append(calculate_score(statement_text, verified_facts))
        json.dump(scores, open(f'{save_dir}/scores_max_{max_debate_round}.json', 'w'))

    if os.path.exists(f'{save_dir}/misleading_max_{max_debate_round}.json'):
        misleading = json.load(open(f'{save_dir}/misleading_max_{max_debate_round}.json'))
    else:
        misleading = detect_misleading(verified_facts, removed_candidate, scores)
        json.dump(misleading, open(f'{save_dir}/misleading_max_{max_debate_round}.json', 'w'))

    trusted_agents = []
    for item in original_all_statements:
        if item['ID'] != int(misleading['statement_idx']):
            trusted_agents.append(item)
    verified_facts = []
    for agent in trusted_agents:
        for statement in agent['statements']:
            verified_facts.append(statement)

    if os.path.exists(f'{save_dir}/{task}_context_facts.json'):
        context_facts = json.load(open(f'{save_dir}/{task}_context_facts.json'))
    else:
        context_facts = extract_facts(shared_information)
        if isinstance(context_facts[0], dict):
            context_facts = [shared_information]
        json.dump(context_facts, open(f'{save_dir}/{task}_context_facts.json', 'w'))

    if os.path.exists(f'{save_dir}/{task}_fact_evaluations_max_{max_debate_round}.json'):
        fact_evaluations = json.load(open(f'{save_dir}/{task}_fact_evaluations_max_{max_debate_round}.json'))
    else:
        fact_evaluations = []
        for fact in context_facts:
            fact_evaluations.append(evaluate_facts(fact, verified_facts))
        json.dump(fact_evaluations, open(f'{save_dir}/{task}_fact_evaluations_max_{max_debate_round}.json', 'w'))
    trustworthy_facts = []
    for item, score in zip(context_facts, fact_evaluations):
        if not isinstance(score, dict):
            continue
        if score['importance'] == 'low':
            trustworthy_facts.append(item)
        elif score['verification'] == 'support':
            trustworthy_facts.append(item)
        else:
            continue

    if os.path.exists(f'{save_dir}/{task}_context_regeneration_max_{max_debate_round}.json'):
        context = json.load(open(f'{save_dir}/{task}_context_regeneration_max_{max_debate_round}.json'))
    else:
        context = context_regeneration(trustworthy_facts + verified_facts)
        json.dump(context, open(f'{save_dir}/{task}_context_regeneration_max_{max_debate_round}.json', 'w'))

    if os.path.exists(f'{save_dir}/{task}_context_decision_max_{max_debate_round}.json'):
        return
    else:
        decision = decision_making(context, question, options)
        json.dump(decision, open(f'{save_dir}/{task}_context_decision_max_{max_debate_round}.json', 'w'))


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
