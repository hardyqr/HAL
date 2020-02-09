from vocab import Vocabulary
import evaluation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='$RUN_PATH/coco_vse/model_best.pth.tar', help='path to model')
parser.add_argument('--data_path', default='data/data', help='path to datasets')
parser.add_argument('--fold5', action='store_true',
                    help='Use fold5')
parser.add_argument('--save_embeddings', action='store_true',
                    help='save_embeddings')
parser.add_argument('--save_csv', default='')

opt_eval = parser.parse_args()

evaluation.evalrank(opt_eval, split='test')
