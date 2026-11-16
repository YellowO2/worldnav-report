"""Parses navigation instructions into a fixed route format with a small language model.

    python 2_features/parse_instructions.py --sample worksheet/instr200.txt
"""
import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'ibm-granite/granite-4.2-3b'
MOVES = {'straight', 'turn', 'curve', 'turn_in_place', 'stairs_up', 'stairs_down'}
SIDES = {'left', 'right', 'none'}
DEGREES = {'slight', 'normal', 'sharp'}

SYSTEM = """You convert a robot navigation instruction into JSON. Output only compact JSON on one line, nothing else.
Format:
{"segments": [{"move": "straight|turn|curve|turn_in_place|stairs_up|stairs_down", "side": "left|right|none",
  "degree": "slight|normal|sharp", "at": "<landmark where the turn happens>" or null,
  "until": "<landmark that ends this segment>" or null}],
 "stop_at": "<landmark where the route stops>" or null}
Rules: one segment per move, in order. A turn is its own segment. "follow the curve" is a curve segment.
Turning to face something without moving is turn_in_place. Ignore landmarks that are only passed and scene
descriptions that are not part of the route (colours of walls, ceilings, sky). Keep landmark phrases short (2-6 words). Use null when not stated.
A landmark is always a thing or a place (a door, the end of the corridor), never an action like "turn right".
Do not repeat the same landmark as both the until of one segment and the until of the next.
stop_at is where the robot ends up. Almost every instruction says this somehow: use the object if one is named
("the glass doors at the end of the hallway"; keep the object together with "end of"), else the place ("end of the
corridor"), else where it is relative to the route ("end of the corridor" when it turns there and stops). null only
when nothing at all is said about where it stops."""

EXAMPLES = [
    ("Walk straight down the hallway, then turn right and stop near the end of the corridor.",
     {"segments": [{"move": "straight", "side": "none", "degree": "normal", "at": None, "until": None},
                   {"move": "turn", "side": "right", "degree": "normal", "at": None, "until": None}],
      "stop_at": "end of the corridor"}),
    ("Walk straight past the wooden partition on your right. Continue forward until you see a patterned chair on your "
     "left, then turn left and stop near the chair.",
     {"segments": [{"move": "straight", "side": "none", "degree": "normal", "at": None, "until": "patterned chair"},
                   {"move": "turn", "side": "left", "degree": "normal", "at": "patterned chair", "until": None}],
      "stop_at": "patterned chair"}),
    ("Start by turning left into a long, curved hallway with large glass windows on the left. Walk straight along this "
     "path, following the curve of the windows. The floor is shiny. Continue forward until you reach the end of this "
     "section, then stop.",
     {"segments": [{"move": "turn", "side": "left", "degree": "normal", "at": None, "until": None},
                   {"move": "straight", "side": "none", "degree": "normal", "at": None, "until": None},
                   {"move": "curve", "side": "none", "degree": "slight", "at": None, "until": "end of this section"}],
      "stop_at": "end of this section"}),
    ("Proceed straight down the long, carpeted hallway with white walls. As you continue forward, you will pass a framed "
     "picture on the right wall. Near the end of the corridor, turn right and stop.",
     {"segments": [{"move": "straight", "side": "none", "degree": "normal", "at": None, "until": "end of the corridor"},
                   {"move": "turn", "side": "right", "degree": "normal", "at": "end of the corridor", "until": None}],
      "stop_at": "end of the corridor"}),
    ("Walk straight down the hallway, passing a recycling bin on the left. Continue until you reach the curved glass "
     "walls at the end of the corridor, then stop.",
     {"segments": [{"move": "straight", "side": "none", "degree": "normal", "at": None,
                    "until": "curved glass walls at the end of the corridor"}],
      "stop_at": "curved glass walls at the end of the corridor"}),
    ("Walk straight along the sidewalk, passing a lamppost on the left. Continue following the path as it curves slightly "
     "right, and stop near the white car.",
     {"segments": [{"move": "straight", "side": "none", "degree": "normal", "at": None, "until": None},
                   {"move": "curve", "side": "right", "degree": "slight", "at": None, "until": "white car"}],
      "stop_at": "white car"}),
]


PREFILL = '{"segments": [{"move": "'     # starting the answer for the model keeps it compact and on format


def prompt(tok, text):
    msgs = [{'role': 'system', 'content': SYSTEM}]
    for q, a in EXAMPLES:
        msgs += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': json.dumps(a)}]
    msgs.append({'role': 'user', 'content': text})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def check(out):
    """The parsed object if it follows the format, else None (out is the text after PREFILL)."""
    m = re.search(r'\{.*\}', out, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d.get('segments'), list) or not d['segments']:
        return None
    # the model sometimes adds a final {"move": "stop"} segment: that is what stop_at is for
    d['segments'] = [s for s in d['segments'] if isinstance(s, dict) and s.get('move') != 'stop']
    if not d['segments']:
        return None
    for s in d['segments']:
        if s.get('move') not in MOVES or s.get('side', 'none') not in SIDES or s.get('degree', 'normal') not in DEGREES:
            return None
        s.setdefault('passing', [])
    return d


@torch.no_grad()
def parse_all(texts, batch=128, max_llm=None):
    """Rule parser first (rule_parse.py)."""
    from rule_parse import rule_parse
    out = [rule_parse(t) for t in texts]
    raw = [None] * len(texts)
    rest = sorted({t for t, o in zip(texts, out) if o is None})
    if max_llm and len(rest) > max_llm:   # parse only a random subset with the language model
        import random
        random.seed(0)
        rest = random.sample(rest, max_llm)
    rest = sorted(rest, key=len)
    print(f'rule parser: {len(texts) - sum(o is None for o in out)}/{len(texts)}; language model: {len(rest)} distinct', flush=True)
    if rest:
        lm_out, lm_raw = parse_llm(rest, batch)
        got = dict(zip(rest, zip(lm_out, lm_raw)))
        for i, t in enumerate(texts):
            if out[i] is None and t in got:
                out[i], raw[i] = got[t]
    return out, raw


@torch.no_grad()
def parse_llm(texts, batch=128):
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to('cuda').eval()
    out = [None] * len(texts)
    raw = [None] * len(texts)
    for attempt, temp in enumerate((0.0, 0.3, 0.6)):
        todo = [i for i, o in enumerate(out) if o is None]
        for b in range(0, len(todo), batch):
            idx = todo[b:b + batch]
            enc = tok([prompt(tok, texts[i]) for i in idx], return_tensors='pt', padding=True).to('cuda')
            gen = model.generate(**enc, max_new_tokens=220, do_sample=temp > 0, temperature=temp or None,
                                 repetition_penalty=1.05, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for i, g in zip(idx, gen[:, enc['input_ids'].shape[1]:]):
                raw[i] = tok.decode(g, skip_special_tokens=True)
                out[i] = check(raw[i])
            print(f'attempt {attempt + 1}: {b + len(idx)}/{len(todo)}', flush=True)
        if attempt == 0:   # batched generation garbles a few that parse fine alone
            for i in [i for i, o in enumerate(out) if o is None]:
                enc = tok([prompt(tok, texts[i])], return_tensors='pt').to('cuda')
                g = model.generate(**enc, max_new_tokens=220, do_sample=False)
                raw[i] = tok.decode(g[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
                out[i] = check(raw[i])
            print('after single retries:', sum(o is None for o in out), 'left', flush=True)
    return out, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('split', nargs='?')
    ap.add_argument('--sample')
    ap.add_argument('--max-llm', type=int, default=None, help='language-model-parse at most this many distinct texts')
    ap.add_argument('--rules-only', action='store_true', help='rule parser only (no GPU); the rest are left empty')
    args = ap.parse_args()
    if args.sample:
        rows = [l.rstrip('\n').split(' ', 2) for l in open(args.sample) if l.strip()]
        texts = [r[2] for r in rows]
        res, raw = parse_all(texts)
        os.makedirs('runs/parse', exist_ok=True)
        with open('runs/parse/sample.jsonl', 'w') as f:
            for (src, style, text), r, w in zip(rows, res, raw):
                f.write(json.dumps(dict(source=src, style=style, instruction=text, parsed=r, raw=None if r else w)) + '\n')
        print('valid', sum(r is not None for r in res), 'of', len(res))
        return
    path = 'data/test/test.jsonl' if args.split == 'test' else f'data/lcvn/{args.split}.jsonl'
    eps = [json.loads(l) for l in open(path)]
    if args.rules_only:
        from rule_parse import rule_parse
        res = [rule_parse(e['instruction']) for e in eps]
        raw = [None] * len(eps)
    else:
        res, raw = parse_all([e['instruction'] for e in eps], max_llm=args.max_llm)
    with open(f'features/{args.split}_parsed.jsonl', 'w') as f:
        for e, r, w in zip(eps, res, raw):
            f.write(json.dumps(dict(episode=e['episode'], parsed=r, raw=None if r else w)) + '\n')
    print('valid', sum(r is not None for r in res), 'of', len(res))


if __name__ == '__main__':
    main()
