"""Writes CodaBench submissions from a trained run.

    python 4_final/submit.py --name v1
"""
import argparse
import json
import zipfile

import torch

from train import Model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--k', type=int, default=6)
    args = ap.parse_args()
    run, dev = f'runs/{args.name}', 'cuda'
    model = Model(k=args.k).to(dev)
    model.load_state_dict(torch.load(f'{run}/model.pt'))
    for zip_name, splits in (('submission_val.zip', ('val_seen', 'val_unseen')), ('submission_test.zip', ('test',))):
        with zipfile.ZipFile(f'{run}/{zip_name}', 'w', zipfile.ZIP_DEFLATED) as z:
            for split in splits:
                preds = predict(model, torch.load(f'features/{split}.pt'), dev)
                doc = {'split': split, 'episodes': [{'episode_id': e, 'actions': a} for e, a in preds.items()]}
                z.writestr(f'{split}.json', json.dumps(doc))
                print(split, len(preds), 'episodes')


if __name__ == '__main__':
    main()
