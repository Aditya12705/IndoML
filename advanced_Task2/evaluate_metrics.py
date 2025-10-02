
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

GT_PATH = 'data/dev_testset.json'
PRED_MI_PATH = 'advanced_Task2/preds_mi.json'
PRED_PG_PATH = 'advanced_Task2/preds_pg.json'

TRACKS = {
    'mi': {
        'pred_path': PRED_MI_PATH,
        'label_key': 'Mistake_Identification',
    },
    'pg': {
        'pred_path': PRED_PG_PATH,
        'label_key': 'Providing_Guidance',
    }
}

def normalize_tutor(name):
    return name.strip().lower()

def get_labels(gt_items, pred_items, label_key):
    gt_labels = {}
    for item in gt_items:
        cid = item['conversation_id']
        for tutor, resp in item.get('tutor_responses', {}).items():
            ann = resp.get('annotation', {})
            if label_key in ann:
                gt_labels[(cid, normalize_tutor(tutor))] = ann[label_key]
    pred_labels = {}
    for item in pred_items:
        cid = item['conversation_id']
        for tutor, resp in item.get('tutor_responses', {}).items():
            ann = resp.get('annotation', {})
            if label_key in ann:
                pred_labels[(cid, normalize_tutor(tutor))] = ann[label_key]
    y_true, y_pred = [], []
    missing = []
    for key in gt_labels:
        if key in pred_labels:
            y_true.append(gt_labels[key])
            y_pred.append(pred_labels[key])
        else:
            missing.append(key)
    return y_true, y_pred, missing

def print_metrics(y_true, y_pred, missing, track):
    print(f'\n=== Metrics for {track.upper()} track ===')
    print(f'Matched pairs: {len(y_true)}')
    if missing:
        print(f'Missing predictions for {len(missing)} pairs (showing up to 5): {missing[:5]}')
    if not y_true or not y_pred:
        print('No matching ground truth and predictions found for this track.')
        return
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1 (macro):', f1_score(y_true, y_pred, average='macro'))
    print('Classification report:')
    print(classification_report(y_true, y_pred))

def main():
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_items = json.load(f)
    for track, info in TRACKS.items():
        with open(info['pred_path'], 'r', encoding='utf-8') as f:
            pred_items = json.load(f)
        y_true, y_pred, missing = get_labels(gt_items, pred_items, info['label_key'])
        print_metrics(y_true, y_pred, missing, track)

if __name__ == '__main__':
    main()
