#!/usr/bin/env python3
"""Render the supervisor .md docs to self-contained .html (base64 figures).
MathJax only for the deep-dive (the others contain literal $ dollar amounts)."""
import re, base64, pathlib, markdown, json, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src.config import provenance

DIR = pathlib.Path("results/supervisor_meeting")
DIR.mkdir(parents=True, exist_ok=True)
(DIR / "provenance.json").write_text(json.dumps(provenance(), indent=2))
CSS = """
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 30px; line-height: 1.55; color: #222; }
  h1 { border-bottom: 2px solid #2c3e50; padding-bottom: 6px; color: #2c3e50; }
  h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; color: #2c3e50; margin-top: 34px; }
  h3 { color: #34495e; margin-top: 24px; }
  table { border-collapse: collapse; margin: 14px 0; font-size: 14px; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #ecf0f1; }
  tr:nth-child(even) { background: #f8f9fa; }
  code { background: #f4f4f4; padding: 1px 5px; border-radius: 3px; font-size: 90%; }
  pre { background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }
  pre code { background: none; }
  img { max-width: 100%; height: auto; display: block; margin: 16px auto;
        border: 1px solid #eee; border-radius: 4px; }
  blockquote { border-left: 4px solid #3498db; margin: 14px 0; padding: 4px 16px;
               background: #f0f7fb; color: #333; }
  hr { border: none; border-top: 1px solid #ddd; margin: 28px 0; }
"""
MATHJAX = ("<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}};</script>"
           "<script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>")

def inline_images(html):
    def repl(m):
        p = DIR / m.group(1)
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return m.group(0)
    return re.sub(r'src="([^"]+\.png)"', repl, html)

def render(name, with_math):
    md = (DIR / f"{name}.md").read_text()
    store = []
    if with_math:
        def protect(m):
            c = m.group(0).replace('<', '\\lt ').replace('>', '\\gt ')
            store.append(c)
            return f"MATHTOKEN{len(store)-1}END"
        md = re.sub(r'\$\$.*?\$\$', protect, md, flags=re.DOTALL)
        md = re.sub(r'\$[^\$\n]+?\$', protect, md)
    body = markdown.markdown(md, extensions=['tables', 'fenced_code', 'sane_lists'])
    for i, c in enumerate(store):
        body = body.replace(f"MATHTOKEN{i}END", c)
    body = inline_images(body)
    head = MATHJAX if with_math else ""
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{name}</title>"
            f"{head}<style>{CSS}</style></head><body>{body}</body></html>")
    (DIR / f"{name}.html").write_text(html)
    print(f"{name}.html written ({len(html)//1024} KB)")

render("SUMMARY", False)
render("CHECKPOINT", False)
render("PHASE_5B_DEEP_DIVE", True)
render("PROJECT_LOG", False)  # formulas in code blocks; $ amounts literal; no MathJax
render("METHODS_WALKTHROUGH", False)
render("ROBUSTNESS_PLAN", False)
