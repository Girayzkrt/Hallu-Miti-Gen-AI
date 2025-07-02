import json
import pandas as pd
from collections import defaultdict
from evaluate import load
import numpy as np
import torch

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def get_bertscore(predictions, references, batch_size=4):
    bertscore = load("bertscore")

    all_precision = []
    all_recall = []
    all_f1 = []

    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i + batch_size]
        batch_refs = references[i:i + batch_size]

        torch.cuda.empty_cache()

        results = bertscore.compute(
            predictions=batch_preds,
            references=batch_refs,
            model_type="microsoft/deberta-xlarge-mnli",
            device="cuda"
        )

        all_precision.extend(results['precision'])
        all_recall.extend(results['recall'])
        all_f1.extend(results['f1'])

    return {
        'precision': all_precision,
        'recall': all_recall,
        'f1': all_f1
    }

def analyze_bertscore(file_path):
    data = load_data(file_path)

    predictions = []
    references = []
    question_type = []

    for item in data:
        predictions.append(item['llm_answer'])
        references.append(item['ground_truth'])
        question_type.append(item['original_question_type'])

    overall_results = get_bertscore(predictions, references)

    type_data = defaultdict(lambda: {'predictions': [], 'references': []})
    for pred, ref, qtype in zip(predictions, references, question_type):
        type_data[qtype]['predictions'].append(pred)
        type_data[qtype]['references'].append(ref)

    type_results = {}
    for qtype, data_dict in type_data.items():
        type_results[qtype] = get_bertscore(
            data_dict['predictions'],
            data_dict['references']
        )

    detailed_results = []
    for i, (pred, ref, qtype) in enumerate(zip(predictions, references, question_type)):
        detailed_results.append({
            'index': i,
            'original_question_type': qtype,
            'ground_truth': ref,
            'llm_answer': pred,
            'precision': overall_results['precision'][i],
            'recall': overall_results['recall'][i],
            'f1': overall_results['f1'][i]
        })

    df = pd.DataFrame(detailed_results)
    df.to_csv(file_path.replace('.jsonl', '_results.csv'), index=False)

    return overall_results, type_results

if __name__ == "__main__":
    file_path = "/content/pure_answers_temp01.jsonl"
    overall_results, type_results = analyze_bertscore(file_path)