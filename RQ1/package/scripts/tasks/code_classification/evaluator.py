

import logging
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            
            
            idx = str(js.get('submission_id', js.get('idx', '')))
            
            
            if 'label' in js:
                
                raw_label = str(js['label']).replace('label_', '')
                target = int(raw_label)
            else:
                target = int(js.get('target', 0))
                
            answers[idx] = target
    return answers

def read_predictions(filename):
    predictions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            
            parts = line.split()
            idx = str(parts[0])     
            label = int(parts[1])   
            predictions[idx] = label
            
    return predictions

def calculate_scores(answers, predictions):
    y_true = []
    y_pred = []
    
    for key in answers:
        if key not in predictions:
            logging.error(f"Missing prediction for index {key}.")
            sys.exit()
        y_true.append(answers[key])
        y_pred.append(predictions[key])

    
    acc = accuracy_score(y_true, y_pred)
    
    
    
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    
    
    
    weight_p, weight_r, weight_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    
    scores = {
        'Accuracy': round(acc, 4),
        'Macro_Precision': round(macro_p, 4),
        'Macro_Recall': round(macro_r, 4),
        'Macro_F1': round(macro_f1, 4),
        'Weighted_Precision': round(weight_p, 4),
        'Weighted_Recall': round(weight_r, 4),
        'Weighted_F1': round(weight_f1, 4)
    }
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions for 250-Class Code Classification dataset.')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in jsonl format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the predictions, in txt format.")
    
    args = parser.parse_args()
    
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    
    scores = calculate_scores(answers, predictions)
    
    
    for metric, value in scores.items():
        print(f"{metric} = {value}")

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()