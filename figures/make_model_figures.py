"""Draws the two model diagrams of section 5 as SVG and PNG (paper style; needs rsvg-convert for the PNG).

    python3 make_model_figures.py   ->  fig5_1_start_model.svg/.png, fig5_2_initial_model.svg/.png
"""
import base64
import json
import math
import os
import subprocess
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO = base64.b64encode(open(os.path.join(HERE, 'example_photo.jpg'), 'rb').read()).decode()
INSTRUCTION = '"Walk straight along the sidewalk, and stop near the group of people ahead."'
ROUTE = json.load(open(os.path.join(HERE, 'example_route.json')))      # 16 (forward, right) points, metres
INK, GREY, FROZEN_FILL = '#000', '#555', '#ececec'


def text(x, y, s, size=13, anchor='middle', color=INK, style='normal'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-style="{style}">{escape(s)}</text>')


def module(x, y, w, h, title, sub=(), frozen=False):
    dash = ' stroke-dasharray="5,3"' if frozen else ''
    fill = FROZEN_FILL if frozen else '#fff'
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{INK}" stroke-width="1.2"{dash}/>']
    lines = [(title, 14, INK)] + [(s, 12, GREY) for s in sub]
    for i, (s, size, color) in enumerate(lines):
        out.append(text(x + w / 2, y + h / 2 + (i - (len(lines) - 1) / 2) * 17 + 5, s, size, color=color))
    return '\n'.join(out)


def arrow(points, label=None, at=-1):
    pts = ' '.join(f'{x},{y}' for x, y in points)
    out = [f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.2" marker-end="url(#a)"/>']
    if label:
        (x1, y1), (x2, y2) = points[at - 1], points[at]
        lines = label.split('\n')
        for i, s in enumerate(lines):
            out.append(text((x1 + x2) / 2, min(y1, y2) - 7 - 13 * (len(lines) - 1 - i), s, 11, color=GREY))
    return '\n'.join(out)


def photo(x, y, w):
    h = w * 225 / 300
    return (f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/jpeg;base64,{PHOTO}"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{INK}" stroke-width="0.8"/>')


def instruction(x, y):
    words, lines, cur = INSTRUCTION.split(), [], ''
    for wd in words:
        if len(cur) + len(wd) > 24:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + ' ' + wd).strip()
    lines.append(cur)
    return '\n'.join(text(x, y + 15 * i, s, 11.5, 'start', GREY, 'italic') for i, s in enumerate(lines))


def route_sketch(x, y, w, h, dots_every=1):
    """The example's true route seen from above: forward = up, right = right."""
    fx = [p[0] for p in ROUTE]
    ry = [p[1] for p in ROUTE]
    scale = min(h / max(max(fx), 1e-6), w / 2 / max(max(abs(v) for v in ry), 0.5))
    cx, base = x + w / 2, y + h
    pts = [(cx, base)] + [(cx + r * scale, base - f * scale) for f, r in zip(fx, ry)]
    out = [f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{INK}" stroke-width="1"/>']
    out += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.2" fill="{INK}"/>' for a, b in pts[1::dots_every]]
    out.append(f'<path d="M{cx - 6},{base + 7} L{cx},{base - 3} L{cx + 6},{base + 7} z" fill="{INK}"/>')
    return '\n'.join(out)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="Arial, Helvetica, sans-serif">\n'
            f'<defs><marker id="a" markerWidth="8" markerHeight="6" refX="7.5" refY="3" orient="auto">'
            f'<path d="M0,0 L8,3 L0,6 z" fill="{INK}"/></marker></defs>\n'
            f'<rect width="{w}" height="{h}" fill="white"/>\n{body}\n</svg>\n')


def start_model():
    b = [photo(20, 20, 150), instruction(20, 168),
         module(260, 30, 170, 64, 'SigLIP2', ['image encoder'], frozen=True),
         module(260, 120, 170, 64, 'MiniLM', ['sentence encoder'], frozen=True),
         module(260, 210, 170, 64, 'SigLIP2', ['text encoder'], frozen=True),
         module(590, 95, 160, 110, 'Transformer', ['encoder-decoder']),
         module(830, 95, 170, 110, 'Per-step head', ['position, heading', 'and stop step']),
         route_sketch(1100, 60, 100, 150),
         arrow([(170, 62), (260, 62)]),
         arrow([(165, 190), (235, 190), (235, 152), (260, 152)]),
         arrow([(165, 190), (235, 190), (235, 242), (260, 242)]),
         arrow([(430, 62), (480, 62), (480, 120), (590, 120)], 'patch vectors'),
         arrow([(430, 152), (590, 152)], 'word vectors'),
         arrow([(430, 242), (480, 242), (480, 184), (590, 184)], 'sentence vector'),
         arrow([(750, 150), (830, 150)]), arrow([(1000, 150), (1085, 150)], 'up to 30\nsteps'),
         text(1150, 240, 'route', 12, color=GREY)]
    return svg(1220, 290, '\n'.join(b))


def initial_model():
    b = [photo(20, 20, 150), instruction(20, 168),
         text(20, 264, 'camera: height, lens, tilt', 11.5, 'start', GREY, 'italic'),
         module(290, 30, 200, 190, 'Qwen3.5-4B', ['vision-language model', 'reads photo + instruction'], frozen=True),
         module(290, 235, 200, 50, 'Linear', []),
         module(680, 95, 180, 150, 'Transformer', ['4 encoder + 2 decoder', 'layers, d = 256']),
         module(940, 130, 120, 80, 'Linear', ['32 outputs']),
         route_sketch(1170, 60, 100, 150),
         arrow([(170, 76), (290, 76)]), arrow([(205, 188), (290, 188)]),
         arrow([(190, 260), (290, 260)]),
         arrow([(490, 125), (530, 125), (530, 145), (680, 145)], '64 cell states\n+ 2 summary states'),
         arrow([(490, 260), (530, 260), (530, 200), (680, 200)], 'camera token'),
         arrow([(860, 170), (940, 170)]), arrow([(1060, 170), (1150, 170)], '16 points'),
         text(1220, 240, 'route, cut into moves', 12, color=GREY),
         text(1220, 256, "at the robot's typical step", 12, color=GREY)]
    return svg(1300, 300, '\n'.join(b))


if __name__ == '__main__':
    for name, fn in (('fig5_1_start_model', start_model), ('fig5_2_initial_model', initial_model)):
        open(os.path.join(HERE, name + '.svg'), 'w').write(fn())
        subprocess.run(['rsvg-convert', '-z', '2', '-o', os.path.join(HERE, name + '.png'),
                        os.path.join(HERE, name + '.svg')], check=True)
