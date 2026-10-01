


import logging
import sys
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            
            
            idx = str(js.get('submission_id', js.get('idx')))
            
            
            if 'status' in js:
                target = 1 if int(js['status']) == 0 else 0
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
            idx, label = line.split()
            predictions[str(idx)] = int(label)
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

    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    
    acc = accuracy_score(y_true, y_pred)
    
    
    p1, r1, f1_1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    p0, r0, f1_0, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], zero_division=0)
    
    
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"Accuracy: {acc:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"Precision_1: {p1[0]:.4f}")
    print(f"Recall_1: {r1[0]:.4f}")
    print(f"F1_1: {f1_1[0]:.4f}")
    print(f"Precision_0: {p0[0]:.4f}")
    print(f"Recall_0: {r0[0]:.4f}")
    print(f"F1_0: {f1_0[0]:.4f}")

    return acc

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Defect Detection dataset.')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the predictions")
    
    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    
    calculate_scores(answers, predictions)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()