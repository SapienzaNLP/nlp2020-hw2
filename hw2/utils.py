import json

def read_dataset(path: str):

    with open(path) as f:
        dataset = json.load(f)

    return dataset


def evaluate_predicate_identification(dataset, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in dataset:
        gold_predicates = dataset[sentence_id]['predicates']
        pred_predicates = predictions[sentence_id]['predicates']
        for g, p in zip(gold_predicates, pred_predicates):
            if g != null_tag and p != null_tag:
                true_positives += 1
            elif p != null_tag and g == null_tag:
                false_positives += 1
            elif g != null_tag and p == null_tag:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_predicate_disambiguation(dataset, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in dataset:
        gold_predicates = dataset[sentence_id]['predicates']
        pred_predicates = predictions[sentence_id]['predicates']
        for g, p in zip(gold_predicates, pred_predicates):
            if g != null_tag and p != null_tag:
                if p == g:
                    true_positives += 1
                else:
                    false_positives += 1
                    false_negatives += 1
            elif p != null_tag and g == null_tag:
                false_positives += 1
            elif g != null_tag and p == null_tag:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_argument_identification(dataset, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in dataset:
        gold_predicates = dataset[sentence_id]['predicates']
        pred_predicates = predictions[sentence_id]['predicates']
        g_idx, p_idx = 0, 0
        for g, p in zip(gold_predicates, pred_predicates):
            if g != null_tag and p == null_tag:
                false_negatives += sum(1 for role in dataset[sentence_id]['roles'][g_idx] if role != null_tag)
            elif g == null_tag and p != null_tag:
                false_positives += sum(1 for role in predictions[sentence_id]['roles'][p_idx] if role != null_tag)
            elif g != null_tag and p != null_tag:
                for r_g, r_p in zip(dataset[sentence_id]['roles'][g_idx], predictions[sentence_id]['roles'][p_idx]):
                    if r_g != null_tag and r_p != null_tag:
                        true_positives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1
            if g != null_tag:
                g_idx += 1
            if p != null_tag:
                p_idx += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_argument_classification(dataset, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in dataset:
        gold_predicates = dataset[sentence_id]['predicates']
        pred_predicates = predictions[sentence_id]['predicates']
        g_idx, p_idx = 0, 0
        for g, p in zip(gold_predicates, pred_predicates):
            if g != null_tag and p == null_tag:
                false_negatives += sum(1 for role in dataset[sentence_id]['roles'][g_idx] if role != null_tag)
            elif g == null_tag and p != null_tag:
                false_positives += sum(1 for role in predictions[sentence_id]['roles'][p_idx] if role != null_tag)
            elif g != null_tag and p != null_tag:
                for r_g, r_p in zip(dataset[sentence_id]['roles'][g_idx], predictions[sentence_id]['roles'][p_idx]):
                    if r_g != null_tag and r_p != null_tag:
                        if r_g == r_p:
                            true_positives += 1
                        else:
                            false_positives += 1
                            false_negatives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1
            if g != null_tag:
                g_idx += 1
            if p != null_tag:
                p_idx += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def _get_table_line(a, b, c):
    if isinstance(b, float):
        b = '{:0.2f}'.format(b)
    if isinstance(c, float):
        c = '{:0.2f}'.format(c)

    line = '{:^20}|{:^20}|{:^20}'.format(a, b, c)
    return line

def print_table(title, results):
    header = _get_table_line('', 'Gold Positive', 'Gold Negative')
    header_sep = '=' * len(header)

    first_line = _get_table_line('Pred Positive', results['true_positives'], results['false_positives'])
    second_line = _get_table_line('Pred Negative', results['false_negatives'], '')

    precision = 'Precision = {:0.4f}'.format(results['precision'])
    recall = 'Recall    = {:0.4f}'.format(results['recall'])
    f1 = 'F1 score  = {:0.4f}'.format(results['f1'])

    output = '{}\n\n{}\n{}\n{}\n{}\n\n\n{}\n{}\n{}\n\n\n'.format(title.upper(), header, header_sep, first_line, second_line, precision, recall, f1)
    return output