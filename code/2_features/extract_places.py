"""Targeted extractor: only where a route stops and where it turns, from any instruction style."""
import re

REACH = (r'(?:until you (?:reach|arrive at|get to|see|are (?:near|at|in front of|next to))|towards?|up to|to reach|'
         r'into|onto)')
ARTICLE = re.compile(r'^(?:the|a|an|this|that|your|its)\s+', re.I)
END = r'(?=,|\.|;|$| and | then | before | where | which | while | as | so | to (?:your|the) (?:left|right)\b)'
STOP_PREP = (r'(?:right in front of|in front of|next to|close to|just before|just past|at the base of|beside|near|by|'
             r'at|under|inside|in|on|past|after|before|beyond)')


def clean(p):
    if not p:
        return None
    p = p.strip(' .,;')
    p = re.sub(r'\s+(?:on|to) (?:your|the) (?:left|right)$', '', p, flags=re.I)
    p = re.sub(r'\s+(?:ahead|in the distance|nearby)$', '', p, flags=re.I)
    p = ARTICLE.sub('', p)
    return p if 0 < len(p.split()) <= 12 else None


def extract(text):
    t = ' '.join(text.split())
    low = t.lower()
    targets = [clean(m.group(1)) for m in re.finditer(
        rf'\b{REACH}\s+(.+?){END}', low)]
    targets = [x for x in targets if x]
    turns = []
    for m in re.finditer(rf'\bturn(?:s|ing)?\s+(?:slightly\s+|sharply\s+)?(left|right)'
                         rf'(?:\s+(?:at|near|by|after|before|past|into|onto|around|when you reach|at the end of)\s+(.+?){END})?', low):
        if re.search(r'\b(?:wall|path|hallway|corridor|road|sidewalk|walkway|it|track|trail)\s+$', low[:m.start()]):
            continue                       # "where the wall turns left": the wall turns, not the robot
        place = clean(m.group(2))
        if place is None:                  # the place reached just before the turn, in the same sentence
            before = low[max(low.rfind('.', 0, m.start()) + 1, 0):m.start()]
            b = re.search(r'(?:\bat|\bnear|once you reach|upon reaching|until you reach|when you reach|until)\s+(.+?)'
                          r'(?:,|\bthen\b|\band\b)\s*(?:then\s+)?$', before)
            if b:
                place = clean(re.sub(r'^(?:the )?point where ', '', b.group(1)))
        turns.append((m.group(1), place))
    stop = None
    for m in re.finditer(rf'\bstop(?:ping)?\s+(?:{STOP_PREP}|(?:when|once|after|upon) (?:you )?(?:reach(?:ing)?|see|arrive at))'
                         rf'\s+(.+?){END}', low):
        stop = clean(m.group(1))                                   # the last stop phrase wins
    if stop in ('them', 'it', 'there', 'those', 'these') and targets:
        stop = targets[-1]                                         # "towards the yellow bollards ... stop near them"
    m = re.search(rf'\bat (?:the )?(end of .+?),? turn(?:s|ing)? (?:left|right)[^.]*?\band stop\b', low)
    if stop is None and m:
        stop = clean(m.group(1))                                   # "at the end of the hallway, turn left and stop"
    if stop is None:                       # no "stop near X": the last place it heads to
        cands = [(m.start(), clean(m.group(1))) for m in re.finditer(
            rf'\b{REACH}\s+(.+?){END}', low)]
        cands += [(m.start(), clean(m.group(1))) for m in re.finditer(
            rf'\bturn(?:s|ing)?\s+(?:left|right)\s+(?:at|near|by|into|onto)\s+(.+?){END}[^.]*?\b(?:and|then) stop\b', low)]
        last_stop = max((m.start() for m in re.finditer(r'\bstop\b', low)), default=len(low))
        cands = [c for c in cands if c[1] and c[0] < last_stop and c[1] not in ('hallway', 'room', 'corridor')]
        if cands:
            stop = max(cands)[1]
    passing = []
    for m in re.finditer(rf'\b(?:passing|past|pass(?:es)?)\s+(.+?){END}', low):
        phr = m.group(1)
        side = re.search(r'\bon (?:your|the) (left|right)\b|\bto (?:your|the) (left|right)\b', low[m.start():m.end() + 25])
        phr = clean(re.sub(r'\s+(?:on|to) (?:your|the) (?:left|right).*$', '', phr))
        if phr and not phr.startswith(('it', 'them', 'this')):
            passing.append((phr, (side.group(1) or side.group(2)) if side else None))
    kind = 'place' if stop else None
    if stop is None and turns and re.search(r'\bturn(?:s|ing)?\s+(?:left|right)[^.]*?\b(?:and|then) stop\b[^.]*\.?\s*$', low):
        kind = 'after the last turn'                               # "... turn left and stop": stops where it turns
        stop = turns[-1][1]
    if stop is None and kind is None and passing and re.search(r'\bstop\b', low[low.rfind(passing[-1][0][:10]):]):
        kind, stop = 'past the last landmark', passing[-1][0]      # "passing the sculpture ..., then stop"
    return dict(stop=stop, stop_kind=kind, turns=turns, targets=targets, passing=passing)


if __name__ == '__main__':
    for s in ["Walk straight down the hallway, passing a yellow bin on the left. Continue forward until you reach the end "
              "of the corridor near the glass windows, then stop.",
              "Walk straight past the high tables and chairs. Turn right towards the large windows and continue straight "
              "until you reach the blue sofa, then stop.",
              "Walk straight through the narrow gap between the brown sofa on the left and the dark furniture on the "
              "right. Continue forward until you reach the end of this passage, then stop.",
              "Walk straight past the kitchen area on the right. Turn left near the end of the carpeted area and stop "
              "facing the patterned structure.",
              "Walk straight along the brick path, keeping the dark tactile strip on your right. Pass the person with the "
              "bicycle and continue towards the white banner hanging overhead."]:
        print(s, '\n   ->', extract(s))
