import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import argparse
import json
import pprint
import requests
import time

from requests.exceptions import ConnectionError
from tqdm import tqdm
from typing import Tuple, List, Any, Dict

import utils


def main(test_path: str, endpoint: str, batch_size=32):

    try:
        dataset = utils.read_dataset(test_path)
    except FileNotFoundError as e:
        logging.error(f'Evaluation crashed because {test_path} does not exist')
        exit(1)
    except Exception as e:
        logging.error(f'Evaluation crashed. Most likely, the file you gave is not in the correct format')
        logging.error(f'Printing error found')
        logging.error(e, exc_info=True)
        exit(1)

    max_try = 10
    iterator = iter(range(max_try))

    while True:

        try:
            i = next(iterator)
        except StopIteration:
            logging.error(f'Impossible to establish a connection to the server even after 10 tries')
            logging.error('The server is not booting and, most likely, you have some error in build_model or StudentClass')
            logging.error('You can find more information inside logs/. Checkout both server.stdout and, most importantly, server.stderr')
            exit(1)

        logging.info(f'Waiting 10 second for server to go up: trial {i}/{max_try}')
        time.sleep(10)

        try:
            response = requests.post(endpoint, json={'data': dataset['0']}).json()
            response['predictions']
            logging.info('Connection succeded')
            break
        except ConnectionError as e:
            continue
        except KeyError as e:
            logging.error(f'Server response in wrong format')
            logging.error(f'Response was: {response}')
            logging.error(e, exc_info=True)
            exit(1)

    predictions = {}

    progress_bar = tqdm(total=len(dataset), desc='Evaluating')

    for sentence_id in dataset:
        sentence = dataset[sentence_id]
        try:
            response = requests.post(endpoint, json={'data': sentence}).json()
            predictions[sentence_id] = response['predictions']
        except KeyError as e:
            logging.error(f'Server response in wrong format')
            logging.error(f'Response was: {response}')
            logging.error(e, exc_info=True)
            exit(1)
        progress_bar.update(1)

    progress_bar.close()

    predicate_identification_results = utils.evaluate_predicate_identification(dataset, predictions)
    predicate_disambiguation_results = utils.evaluate_predicate_disambiguation(dataset, predictions)
    argument_identification_results = utils.evaluate_argument_identification(dataset, predictions)
    argument_classification_results = utils.evaluate_argument_classification(dataset, predictions)

    print(utils.print_table('predicate identification', predicate_identification_results))
    print(utils.print_table('predicate disambiguation', predicate_disambiguation_results))
    print(utils.print_table('argument identification', argument_identification_results))
    print(utils.print_table('argument classification', argument_classification_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help='File containing data you want to evaluate upon')
    args = parser.parse_args()

    main(
        test_path=args.file,
        endpoint='http://127.0.0.1:12345'
    )
