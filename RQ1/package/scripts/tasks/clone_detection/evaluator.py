


import logging
import sys
import json
import numpy as np

def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js['index']] = js['answers']
    return answers

def read_predictions(filename):
    predictions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            predictions[js['index']] = js['answers']
    return predictions

def calculate_scores(answers, predictions):
    map_scores = []
    mrr_scores = []
    p_at_1 = []
    p_at_5 = []

    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
            
        if len(predictions[key]) < len(answers[key]):
            logging.error("Not enough predictions for index {}: {} < {}.".format(
                key, len(predictions[key]), len(answers[key])))
            sys.exit()

        answer_set = set(answers[key])
        preds = predictions[key][:len(answers[key])]

        
        Avep = []
        for k, p in enumerate(preds):
            if p in answer_set:
                Avep.append((len(Avep) + 1) / (k + 1))
        
        if len(answer_set) > 0:
            map_scores.append(sum(Avep) / len(answer_set))
        else:
            map_scores.append(0.0)

        
        mrr = 0.0
        for k, p in enumerate(preds):
            if p in answer_set:
                mrr = 1.0 / (k + 1)
                break
        mrr_scores.append(mrr)

        
        def get_precision_at_k(k_val):
            top_k = preds[:k_val]
            if not top_k:
                return 0.0
            hits = sum(1 for p in top_k if p in answer_set)
            return hits / len(top_k)

        
        p_at_1.append(get_precision_at_k(1))
        p_at_5.append(get_precision_at_k(5))

    
    result = {
        'MAP@R': round(np.mean(map_scores), 4),
        'MRR': round(np.mean(mrr_scores), 4),
        'P@1': round(np.mean(p_at_1), 4),
        'P@5': round(np.mean(p_at_5), 4)
    }
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for POJ-104 dataset.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    
    scores = calculate_scores(answers, predictions)
    
    
    for metric, value in scores.items():
        print(f"{metric} = {value}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()