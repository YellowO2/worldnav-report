"""Builds report.pdf from report.md (A4, single column, Arial 10 pt) using Google Chrome to print.

    python3 build_pdf.py   ->  report.pdf (and prints the page count)
"""
import html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', 'google-chrome', 'chromium']

CSS = '''
@page { size: A4; margin: 2cm; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 10pt; line-height: 1.35; color: #000; orphans: 3; widows: 3; }
h1, h2, h3, h4 { break-after: avoid; page-break-after: avoid; line-height: 1.2; }
h2 { font-size: 13pt; margin: 1.4em 0 0.5em; }
h3 { font-size: 11pt; margin: 1.1em 0 0.4em; }
h4 { font-size: 10pt; margin: 0.9em 0 0.3em; }
h2.newpage { page-break-before: always; margin-top: 0; }
p { margin: 0 0 0.6em; text-align: justify; hyphens: auto; }
ul, ol { margin: 0.2em 0 0.6em; padding-left: 1.4em; }
li { margin: 0.15em 0; text-align: left; }
table { border-collapse: collapse; margin: 0.8em auto 0.3em; font-size: 9pt; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3px 6px; vertical-align: top; text-align: left; }
th { background: #eee; font-weight: bold; }
img { display: block; max-width: 100%; margin: 0.8em auto 0.3em; page-break-inside: avoid; }
img[src*="figA1"] { width: 70%; }
img[src*="fig6_1"] { width: 65%; }
a { color: inherit; text-decoration: none; }
a[href^="http"] { color: #1a4f9c; }
p.caption { text-align: center; font-size: 9pt; margin: 0.2em 0 1em; page-break-before: avoid; }
pre { font-family: Arial, Helvetica, sans-serif; text-align: center; margin: 0.6em 0; }
.cover { display: flex; flex-direction: column; align-items: center; text-align: center; height: 25.4cm; justify-content: center; page-break-after: always; }
.cover img { width: 65%; }
.cover .title { font-size: 20pt; font-weight: bold; line-height: 1.3; margin: 2.5cm 0 1cm; max-width: 16.5cm; }
.cover .info { font-size: 13pt; line-height: 1.8; margin: 0; text-align: center; }
.cover .info b { font-size: 14pt; }
.cover table { margin: 1.5cm 0 0; width: 14cm; font-size: 12pt; border-collapse: collapse; }
.cover th, .cover td { border: 1px solid #000; padding: 4px 10px; text-align: center; }
.cover th { background: #e6e6e6; }
'''


def ref_id(kind, num):
    return ('fig-' if kind == 'Figure' else 'tab-') + num.replace('.', '-')


def inline(t, caption=False):
    t = html.escape(t)
    t = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', t)
    t = re.sub(r'(https?://[^\s<]+[^\s<.,;)])', r'<a href="\1">\1</a>', t)
    if not caption and '<img' not in t:   # 'Figure 3', 'Table A.1' in the text jump to that figure or table
        t = re.sub(r'\b(Figure|Table) ([A-Z]?\.?[0-9]+)\b', lambda m: f'<a href="#{ref_id(m[1], m[2])}">{m[0]}</a>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    return t


def to_html(md):
    out, lines, i = [], md.split('\n'), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line[level:].strip()
            cls = ' class="newpage"' if level == 2 and text.startswith(('References', 'Appendix')) else ''
            out.append(f'<h{level}{cls}>{inline(text)}</h{level}>')
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            body = [r for r in rows if not all(re.fullmatch(r':?-+:?', c) for c in r)]
            out.append('<table>' + '<tr>' + ''.join(f'<th>{inline(c)}</th>' for c in body[0]) + '</tr>'
                       + ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in body[1:])
                       + '</table>')
        elif line.startswith('    '):
            block = []
            while i < len(lines) and (lines[i].startswith('    ') or not lines[i].strip()):
                block.append(lines[i][4:])
                i += 1
            out.append('<pre>' + html.escape('\n'.join(block).strip('\n')) + '</pre>')
        elif re.match(r'\s*(- |\d+\. )', line):
            ordered = bool(re.match(r'\s*\d+\. ', line))
            items = []
            while i < len(lines) and re.match(r'\s*(- |\d+\. )', lines[i]):
                items.append(re.sub(r'^\s*(- |\d+\. )', '', lines[i]))
                i += 1
            tag = 'ol' if ordered else 'ul'
            out.append(f'<{tag}>' + ''.join(f'<li>{inline(x)}</li>' for x in items) + f'</{tag}>')
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r'(#|\||    |\s*(- |\d+\. ))', lines[i]):
                para.append(lines[i].strip())
                i += 1
            m = re.match(r'(Table|Figure) ([A-Z]?\.?[0-9]+):', para[0]) if len(para) == 1 else None
            if m:
                out.append(f'<p class="caption" id="{ref_id(m[1], m[2])}">{inline(para[0], caption=True)}</p>')
            else:
                out.append('<p>' + '<br>'.join(inline(x) for x in para) + '</p>')
    return '\n'.join(out)


def cover_html(lines):
    """Cover page from the lines above the first section: '# title', course lines, then Name / Matriculation pairs."""
    lines = [x.strip() for x in lines if x.strip() and not x.strip().startswith('(Contents')]
    k = lines.index('Name') if 'Name' in lines else len(lines)
    top, people = lines[:k], lines[k + 2:]
    title = next((x[2:] for x in top if x.startswith('# ')), '')
    rest = [x for x in top if not x.startswith('# ')]
    info = rest
    e = html.escape
    out = ['<div class="cover"><img src="figures/ntu_logo.png" alt="NTU">',
           f'<div class="title">{e(title)}</div>',
           '<p class="info">' + '<br>'.join(f'<b>{e(x)}</b>' if 'SC4001' in x else e(x) for x in info) + '</p>']
    if people:
        out.append('<table><tr><th>Name</th><th>Matriculation Number</th></tr>')
        out += [f'<tr><td>{e(a)}</td><td>{e(b)}</td></tr>' for a, b in zip(people[::2], people[1::2])]
        out.append('</table>')
    return ''.join(out) + '</div>'


def main():
    md = open(os.path.join(HERE, 'report.md')).read()
    lines = md.split('\n')
    first = next(i for i, x in enumerate(lines) if x.startswith('## '))
    body = cover_html(lines[:first]) + to_html('\n'.join(lines[first:]))
    page = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    html_path, pdf_path = os.path.join(HERE, '_report_build.html'), os.path.join(HERE, 'report.pdf')
    open(html_path, 'w').write(page)
    for chrome in CHROME:
        try:
            subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                            f'--print-to-pdf={pdf_path}', 'file://' + html_path],
                           check=True, capture_output=True, timeout=120)
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    else:
        sys.exit('Google Chrome not found: open report.html in a browser and print it to PDF instead')
    os.remove(html_path)
    pages = len(re.findall(rb'/Type\s*/Page[^s]', open(pdf_path, 'rb').read()))
    print(f'report.pdf: {pages} pages')


if __name__ == '__main__':
    main()
