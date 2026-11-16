"""One parse per route, from its concise and landmark-style instructions together.

    python 2_features/build_parses.py train|val_seen|val_unseen|test
"""
import collections
import hashlib
import json
import re
import sys

from extract_places import extract
from rule_parse import rule_parse

PLACE = {'end', 'corridor', 'path', 'hallway', 'room', 'passage', 'walkway', 'sidewalk', 'road', 'this', 'section',
         'far', 'the', 'of', 'a', 'straight', 'segment', 'area', 'trajectory'}


def stop_kind(stop, kind):
    if kind == 'after the last turn':
        return 'after turn'
    if kind == 'past the last landmark':
        return 'past landmark'
    if not stop:
        return 'none'
    return 'end' if not (set(re.findall(r'[a-z]+', stop.lower())) - PLACE) else 'object'


def merge(concise_text, landmark_text, saved):
    base = rule_parse(concise_text) or saved
    eL, eC = extract(landmark_text), extract(concise_text)
    if base is None:                                 # neither parser: the landmark style's turns, straight between
        segs = [dict(move='straight', side='none', degree='normal', at=None, until=None, passing=[])]
        for side, place in eL['turns']:
            segs.append(dict(move='turn', side=side, degree='normal', at=place, until=None, passing=[]))
        base = dict(segments=segs, stop_at=None)
    segs = [dict(s) for s in base['segments']]
    pool = list(eL['turns']) + list(eC['turns'])
    for s in segs:                                   # turn places, matched in order on the same side
        if s['move'] in ('turn', 'curve') and not s.get('at'):
            for k, (side, place) in enumerate(pool):
                if side == s.get('side') and place:
                    s['at'] = place
                    pool = pool[k + 1:]
                    break
    stop, kind = None, None
    for e in (eL, eC):
        if e['stop'] and e['stop_kind'] == 'place':
            stop, kind = e['stop'], 'place'
            break
    if stop is None and base.get('stop_at'):
        stop, kind = base['stop_at'], 'place'
    if stop is None:
        for e in (eL, eC):
            if e['stop_kind']:
                stop, kind = e['stop'], e['stop_kind']
                break
    if segs and segs[-1]['move'] == 'straight' and segs[-1].get('until') and stop and \
            segs[-1]['until'].lower() == stop.lower():
        segs[-1]['until'] = None                     # the stop slot already carries it
    return dict(segments=segs, stop_at=stop), stop_kind(stop, kind), eL['passing']


def main(split):
    path = 'data/test/test.jsonl' if split == 'test' else f'data/lcvn/{split}.jsonl'
    eps = [json.loads(line) for line in open(path)]
    saved = {json.loads(l)['episode']: json.loads(l)['parsed'] for l in open(f'features/{split}_parsed.jsonl')}
    routes = collections.defaultdict(list)
    for e in eps:
        key = hashlib.md5(open(e['image'], 'rb').read()).hexdigest() if split == 'test' else e['episode'][:-2]
        routes[key].append(e)
    rows, kinds = {}, collections.Counter()
    for key, group in routes.items():
        if len(group) == 3:
            order = sorted(group, key=lambda e: len(e['instruction'])) if split == 'test' else \
                sorted(group, key=lambda e: {'0': 0, '2': 1, '1': 2}[e['episode'][-1]])
            parsed, kind, passing = merge(order[0]['instruction'], order[1]['instruction'], saved.get(order[0]['episode']))
            for e in group:
                rows[e['episode']] = (parsed, kind, passing)
            kinds[kind] += 1
        else:                                        # not one route in three styles: each episode on its own
            for e in group:
                parsed, kind, passing = merge(e['instruction'], e['instruction'], saved.get(e['episode']))
                rows[e['episode']] = (parsed, kind, passing)
                kinds[kind] += 1
    with open(f'features/{split}_parsed2.jsonl', 'w') as f:
        for e in eps:
            parsed, kind, passing = rows[e['episode']]
            f.write(json.dumps(dict(episode=e['episode'], parsed=parsed, stop_kind=kind, passing=passing)) + '\n')
    n = sum(kinds.values())
    print(f'{split}: {n} routes; where it stops:', {k: f'{v} ({v / n:.0%})' for k, v in kinds.most_common()})


if __name__ == '__main__':
    main(sys.argv[1])
