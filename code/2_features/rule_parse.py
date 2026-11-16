"""Rule parser for the templated (mostly concise-style) instructions."""
import re

MOVE_VERB = r'(?:walk|move|go|proceed|head|continue|keep going|keep walking|drive)'
DIRWORD = r'(?:straight(?: ahead| forward)?|forward|ahead|straight on)'
PREP = r'(?:down|along|across|through|past|towards?|into|onto|on|over|up|by|beside|between|around|under)'
STOP_PREP = r'(?:near|at|in front of|by|beside|next to|close to|under|on|inside|in|right in front of|just before|before|after|past|beyond)'
SIDE = r'(left|right)'
ARTICLE = re.compile(r'^(?:the|a|an|this|that|your)\s+', re.I)


def np_(text):
    """A short landmark phrase: articles and trailing clutter removed."""
    t = text.strip(' .,;')
    t = re.sub(r'\s+(?:ahead|in the distance|on (?:the|your) (?:left|right)|to (?:the|your) (?:left|right))$', '', t, flags=re.I)
    t = ARTICLE.sub('', t)
    return t if 0 < len(t.split()) <= 12 else None


def seg(move, side='none', at=None, until=None, passing=None, degree='normal'):
    return dict(move=move, side=side, degree=degree, at=at, until=until, passing=passing or [])


def clause(c):
    """One clause -> (list of segments, stop landmark or '' for a bare stop, or None for no stop), or None."""
    c = c.strip(' .,;').lower()
    c = re.sub(r'^(?:then|and|and then|finally|after that)\s+', '', c)
    if not c:
        return [], None
    m = re.fullmatch(rf'(?:and\s+)?stop(?:\s+{STOP_PREP}\s+(.+))?', c)
    if m:
        return [], (np_(m.group(1)) if m.group(1) else '')
    m = re.fullmatch(rf'turn {SIDE}(?: (?:at|into|onto|towards?|near|around|past|before|after) (.+))?', c)
    if m and m.group(2) and re.search(r'\b(?:and|follow|then|continue|walk)\b', m.group(2)):
        return None                                       # more than a turn in this clause: leave it to the model
    if m:
        return [seg('turn', m.group(1), at=np_(m.group(2)) if m.group(2) else None)], None
    m = re.fullmatch(rf'turn slightly {SIDE}(?: .+)?', c)
    if m:
        return [seg('turn', m.group(1), degree='slight')], None
    m = re.fullmatch(rf'(?:{MOVE_VERB} )?(?:{DIRWORD} )?until you reach (.+)', c)
    if m:
        return [seg('straight', until=np_(m.group(1)))], None
    m = re.fullmatch(rf'{MOVE_VERB}(?: {DIRWORD})?(?: ({PREP}) (.+?))?(?: until you reach (.+))?', c)
    if m:
        until = np_(m.group(3)) if m.group(3) else None
        where = m.group(2) or ''
        if until is None and m.group(1) in ('toward', 'towards'):
            until = np_(where)                            # "walk towards X": X is where this stretch heads
        elif until is None and re.search(r'\btowards? ', where):
            until = np_(re.split(r'\btowards? ', where, maxsplit=1)[1])   # "across the field towards the trees"
        along = np_(where) if m.group(1) in ('along', 'down', 'through', 'across') and not until else None
        return [seg('straight', until=until, at=along)], None
    m = re.fullmatch(rf'(?:{MOVE_VERB} )?(?:{DIRWORD} )?towards? (.+)', c)
    if m:
        return [seg('straight', until=np_(m.group(1)))], None
    return None


def rule_parse(text):
    """The parsed route, or None when any clause falls outside the patterns."""
    parts = re.split(r'(?<=[.!?])\s+|,\s*|\s+(?:and then|then|and)\s+(?=(?:turn|stop|walk|move|go|proceed|continue|head)\b)',
                     text.strip(), flags=re.I)
    segments, stop = [], None
    for p in parts:
        r = clause(p)
        if r is None:
            return None
        s, st = r
        if segments and s and s[0]['move'] == 'straight' and segments[-1]['move'] == 'straight' and not segments[-1]['until']:
            segments[-1]['until'] = s[0]['until']          # "walk straight..., continue until X": one straight stretch
        else:
            segments += s
        if st is not None:
            stop = st or None
            break
    if not segments:
        return None
    if stop and segments[-1]['move'] == 'straight' and not segments[-1]['until']:
        segments[-1]['until'] = stop
    if not stop and segments[-1]['move'] == 'straight' and segments[-1]['until']:
        stop = segments[-1]['until']                      # "... until you reach X, then stop": stop at X
    return dict(segments=segments, stop_at=stop)


if __name__ == '__main__':
    for t in ["Walk straight along the hedge, turn right into the alley, and stop near the metal building.",
              "Turn left and walk straight across the grassy field, then stop near the parked vehicles.",
              "Walk straight, turn left, then turn right and stop.",
              "Move forward and turn right, then stop near the desk.",
              "Walk straight through the office area. Continue forward until you reach the glass-walled room on the left, then stop.",
              "Walk straight down the hallway and stop near the end.",
              "Walk straight past the yellow beanbag chair and the blue sofa. Turn right near the large windows and stop.",
              "Walk straight ahead across the grass, following the curve of the path."]:
        print(t, '\n   ->', rule_parse(t))
