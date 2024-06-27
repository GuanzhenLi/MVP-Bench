import os
import sys
import json
import codecs
import jsonlines
import argparse
from tqdm import tqdm

import sys
from vlmeval.evaluate.misc import build_judge
from vlmeval.evaluate.vqa_eval import process_line
from vlmeval.evaluate.coco_eval import COCO_Caption_Scorer

os.environ['OPENAI_API_KEY'] = "Your_Openai_Key"
os.environ['GOOGLE_API_KEY'] = 'Your_Google_Key'

def jsonline_load(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [r for r in reader]
    return data

def jsonlines_dump(data, fname):
    with jsonlines.open(fname, mode='a' if os.path.exists(fname) else 'w') as writer:
        writer.write_all(data)

def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def yes_or_no_eval(model, question, prediction, answer):
    
    def extract_yn(output):
        s = output.lower()
        words = process_punctuation(s).split()
        if 'yes' in words and 'no' not in words:
            return 'Yes'
        if 'yes' not in words and 'no' in words:
            return 'No'
        return 'Unknown'
    
    tmpl = (
        'You are an AI assistant who will help me to match an answer with two options of a question. '
        'The options are only Yes / No. '
        'You are provided with a question and an answer, '
        'and you need to find which option (Yes / No) is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Unknown. '
        'Your should output a single word among the following 3 choices: Yes, No, Unknown.\n'
        'Example 1: \n'
        "Question: Is the word in this image 'Hello'?\nAnswer: The word in this image is 'Hello'.\nYour output: Yes\n"
        'Example 2: \n'
        "Question: Is the word in this image 'Hello'?\n"
        "Answer: The word in this image is not 'Hello'.\nYour output: No\n"
        'Example 3: \n'
        "Question: Is the word in this image 'Hello'?\nAnswer: Yes.\nYour output: Yes\n"
        'Example 4: \n'
        "Question: Is the word in this image 'Hello'?\n"
        "Answer: No, the word in this image is not 'Hello'.\nYour output: No\n"
        'Example 5: \n'
        'Question: {}?\nAnswer: {}\nYour output: '
    )
    prompt = tmpl.format(question, prediction)
    retry = 5
    ans = 'Unknown'
    for i in range(retry):
        output = model.generate(prompt, temperature=0.5 * i)
        ans = extract_yn(output)
    
    return ans

def extract_predictions(predictions, model_name, judge_type='chatgpt-1106'):
    judge_kwargs = dict(model=judge_type, verbose=True)
    judge = build_judge(**judge_kwargs)
    
    eval_results = []
    if os.path.exists(os.path.join(OUTPUT_DIR, f'{model_name}_evaluated.jsonl')):
        eval_results = jsonline_load(os.path.join(OUTPUT_DIR, f'{model_name}_evaluated.jsonl'))
    existed_ids = [x['question_id'] for x in eval_results]
    for dt in tqdm(predictions):
        qid = dt['question_id']
        if qid in existed_ids: continue
        pred = dt['output'] if 'output' in dt else dt['text']
        sample = qas[qid]
        data = {
            'question_id': qid,
            'image': sample['image'],
            'question': sample['question'],
            'prediction': pred,
            'answer': sample['answer'],
            'type': sample['type'],
            'level': sample['level']
        }
        if sample['type'].startswith('y/n'):
            data.update({
                'extracted (rule)': pred.split('.')[0].strip().split(',')[0].strip(),
                'extracted (GPT)': yes_or_no_eval(judge, sample['question'], pred, sample['answer']) if pred.split('.')[0].strip().split(',')[0].strip().lower() not in ['yes', 'no'] else None,
            })
        elif sample['type'].startswith('mcq'):
            data.update({
                'extracted (rule)': pred.split('.')[0].strip().split(',')[0].strip().split(')')[0].strip('()').strip()
            })
        eval_results.append(data)
        jsonlines_dump(eval_results[-1:], os.path.join(OUTPUT_DIR, f'{model_name}_evaluated.jsonl'))
        existed_ids.append(qid)    
    return eval_results

def print_out_accuracy(accuracy, qAcc, iAcc, mAcc):
    for k, v in accuracy.items():
        if k == 'fb': continue 
        cnt = {}
        for kk, vv in v.items():
            cnt[kk] = []
            for kkk, vvv in vv.items():
                print('-------- ({}) {}'.format(kkk, sum(vvv)/len(vvv)))
                cnt[kk].extend(vvv)
            print('==== [{}] {}'.format(kk, sum(cnt[kk])/max(1, len(cnt[kk]))))
        cntx = []
        for vv in cnt.values():
            cntx.extend(vv)
        print('## {}: {}'.format(k, sum(cntx)/len(cntx)))
    
    print('=======')
    for t, Acc in zip(('qAcc', 'iAcc', 'mAcc'), (qAcc, iAcc, mAcc)):
        print("ACC")
        if t == 'qAcc':
            low = [all(x['correct']) for x in Acc.values() if x['level'] == 'low']
            high = [all(x['correct']) for x in Acc.values() if x['level'] == 'high']
            all_ = [all(x['correct']) for x in Acc.values()]
            print(t, sum(all_)/len(all_), sum(low)/max(1, len(low)), sum(high)/max(len(high), 1), len(low), len(high))
        elif t == 'iAcc':
            low = [all(x) for k,x in Acc.items() if k.count('_s') if len(x) > 1]
            high = [all(x) for k,x in Acc.items() if k.count('_e') if len(x) > 1]
            all_ = [all(x) for x in Acc.values() if len(x) > 1]
            print(t, sum(all_)/len(all_), sum(low)/max(1, len(low)), sum(high)/max(len(high), 1), len(low), len(high))
        else:
            all_ = [all(x) for x in Acc.values() if len(x) > 3]
            print(t, sum(all_)/len(all_), len(all_))
    
def calculate_overall_accuracy(eval_results):
    accuracy = {
        'y/n': {'low': {}, 'high': {}}, 'mcq': {'low': {}, 'high': {}}, 'fb': {'low': {}, 'high': {}},
    }
    qAcc, iAcc, mAcc = {}, {}, {}
    for sample in tqdm(eval_results):
        qid = sample['question_id']
        pred = sample['prediction']
        
        dt_id = int(sample['image'].split('/')[-1].split('_')[0])
        img_id = sample['image']
        qu = sample['question']
        
        if sample['type'].startswith('y/n'):
            if sample['type'] not in accuracy['y/n'][sample['level']]:
                accuracy['y/n'][sample['level']][sample['type']] = []
                accuracy['y/n'][sample['level']][sample['type']] = []
            pred = sample['extracted (rule)'] if sample['extracted (rule)'].lower() in ['yes', 'no'] else sample['extracted (GPT)']
            is_correct = pred.lower() == sample['answer'].lower()
            accuracy['y/n'][sample['level']][sample['type']].append(is_correct)
            if (dt_id, qu) not in qAcc: qAcc[(dt_id, qu)] = {'level': sample['level'], 'correct': []}
            if img_id not in iAcc: iAcc[img_id] = []
            if dt_id not in mAcc: mAcc[dt_id] = []
            qAcc[(dt_id, qu)]['correct'].append(is_correct)
            iAcc[img_id].append(is_correct)
            mAcc[dt_id].append(is_correct)
        elif sample['type'].startswith('mcq'):
            if sample['type'] not in accuracy['mcq'][sample['level']]:
                accuracy['mcq'][sample['level']][sample['type']] = []
            accuracy['mcq'][sample['level']][sample['type']].append(sample['extracted (rule)'].lower() == sample['answer'].lower())
            # if sample['extracted (rule)'] not in 'ABCDE':
            #     import ipdb; ipdb.set_trace()
        else:
            if sample['type'] not in accuracy['fb'][sample['level']]:
                accuracy['fb'][sample['level']][sample['type']] = []
            if ' is ' in pred and ' is _' in sample['question']:
                pred = pred.split(' is ')[-1].strip()
            elif ' are ' in pred and ' are _' in sample['question']:
                pred = pred.split(' are ')[-1].strip()
            accuracy['fb'][sample['level']][sample['type']].append((pred.lower(), sample['answer'].lower()))
    return accuracy, qAcc, iAcc, mAcc

def evaluate_mcq(fname, mcq_qa):
    qas = {x['question_id']:x for x in jsonline_load(mcq_qa)}
    predictions = jsonline_load(fname)
    eval_results = {'low': {'c': {}, 'e': {}}, 'high': {'c': {}, 'e': {}}}
    for prd in predictions:
        t = 'e' if '_e' in qas[prd['question_id']]['image'] else 'c'
        level = qas[prd['question_id']]['level']
        mcq_id = qas[prd['question_id']]['mcq_id']
        if mcq_id not in eval_results[level][t]: eval_results[level][t][mcq_id] = []
        eval_results[level][t][mcq_id].append(prd['output'].strip().split('.')[0].strip().split(',')[0].strip() == qas[prd['question_id']]['answer'])
    for k, v in eval_results.items():
        for kk, vv in v.items():
            accu = [all(x) for x in vv.values()]
            print(k, f'mcq-{kk}', sum(accu)/max(1, len(accu)), len(accu))
            accu = [x[0] for x in vv.values()]
            print(k, f'mcq-{kk}', sum(accu)/max(1, len(accu)), len(accu))

if __name__ == '__main__':
    
    ###=== Calculate Overall Accuracy ===###
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default='GPT4o',type=str)
    parser.add_argument("--input_dir", default="model_prediction", type=str)
    parser.add_argument("--output_dir", default='model_prediction', type=str)
    parser.add_argument("--qa_dir", default="data/all_questions.json", type=str)
    parser.add_argument("--mcq_dir", default="data/mcq_questions.json", type=str)
    args = parser.parse_args()

    model_name = args.model_name
    OUTPUT_DIR = args.output_dir
    qas = jsonline_load(args.qa_dir)
    qas = {x['question_id']:x for x in qas}
    
    predictions = jsonline_load(os.path.join(OUTPUT_DIR, f'{model_name}.jsonl'))
    eval_results = extract_predictions(predictions, model_name, judge_type='chatgpt-1106')
    accuracy, qAcc, iAcc, mAcc = calculate_overall_accuracy(eval_results)
    print_out_accuracy(accuracy, qAcc, iAcc, mAcc)
    evaluate_mcq(os.path.join(OUTPUT_DIR, f'mcq_{model_name}.jsonl'), args.mcq_dir)