import os
import json
import codecs
import jsonlines
from tqdm import tqdm

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
# sys.path.append('/home/users/nus/e0672129/scratch/VLMEvalKit')
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

os.environ['OPENAI_API_KEY'] = "Your_Openai_Key"
os.environ['GOOGLE_API_KEY'] = 'Your_Google_Key'

def jsonline_load(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [r for r in reader]
    return data

def jsonlines_dump(data, fname):
    with jsonlines.open(fname, mode='w') as writer:
        writer.write_all(data)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='GPT4o',type=str)
    parser.add_argument("--system_prompt", default='', type=str)
    parser.add_argument("--img_dir", default='data/Images', type=str)
    parser.add_argument("--output_dir", default='model_prediction', type=str)
    parser.add_argument("--qas_pth", default='data/all_questions.json', type=str)
    parser.add_argument("--mcq_pth", default='data/mcq_all_questions.json', type=str)
    parser.add_argument("--question_type", default="all_questions", type=str) # all_questions/mcq_questions
    args = parser.parse_args()
    
    if args.question_type == 'all_questions':
        qas = jsonline_load(args.qas_pth)
    if args.question_type == 'mcq_questions':
        qas = jsonline_load(args.mcq_pth)
    model_name = args.model_name
    OUTPUT_DIR = args.output_dir
    IMG_DIR = args.img_dir
    system_prompt = args.system_prompt
    
    output_path = os.path.join(OUTPUT_DIR, f'{model_name}_supp.pkl')
    
    if not system_prompt:
        model = supported_VLM[model_name]()
    else:
        model = supported_VLM[model_name](system_prompt=system_prompt)
    gen_func = model.generate
    
    res = {}
    if os.path.exists(output_path):
        res = load(output_path)
    
    indices = [struct['question_id'] for struct in qas if struct['question_id'] not in res]
    
    if not system_prompt:
        structs = [dict(message=[
            {'type': 'image', 'value': os.path.join(IMG_DIR, struct['image'])},
            {'type': 'text', 'value': system_prompt + '\n' + struct['question']},
        ], dataset='PerceptionBench') for struct in qas if struct['question_id'] not in res]
    else:
        structs = [dict(message=[
            {'type': 'image', 'value': os.path.join(IMG_DIR, struct['image'])},
            {'type': 'text', 'value': struct['question']},
        ], dataset='PerceptionBench') for struct in qas if struct['question_id'] not in res]
    
    if model_name in ['GPT4V', 'GPT4o']:
        if len(structs):
            track_progress_rich(gen_func, structs, nproc=4, chunksize=4, save=output_path, keys=indices)
    else:
        for i, struct in tqdm(zip(indices, structs)):
            response = model.generate(**struct)
            torch.cuda.empty_cache()
            # torch.cuda("1")
            res[i] = response
            if (len(res) + 1) % 20 == 0:
                dump(res, output_path)
        dump(res, output_path)
    
    res = load(output_path)
    
    res = {k: {'question_id': k, 'output': v} for k,v in res.items()}
    for x in qas:
        res[x['question_id']].update({
            'prompt': x['question'],
            'image': x['image'],
            'model_id': model_name
        })
        
    if args.question_type == 'all_questions':
        jsonlines_dump(res.values(), os.path.join(OUTPUT_DIR, f'{model_name}.jsonl'))
    if args.question_type == 'mcq_questions':
        jsonlines_dump(res.values(), os.path.join(OUTPUT_DIR, f'mcq_{model_name}.jsonl'))
    
    os.remove(output_path)
    