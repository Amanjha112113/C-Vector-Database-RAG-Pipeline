import re

with open('python_app/static/index.html', 'r') as f:
    code = f.read()

new_col = """
let COL = {};
function updateColors() {
  const get = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  COL = {
    cs: get('--cs'), math: get('--math'), food: get('--food'), sports: get('--sports'),
    doc: get('--green'), default: get('--muted'), bg: get('--bg'), border: get('--border'),
    text: get('--text'), muted: get('--muted'), accent: get('--accent'),
    accentGlow: get('--accent-glow')
  };
}
function getDimCol(i) {
  if(i<4) return COL.cs;
  if(i<8) return COL.math;
  if(i<12) return COL.food;
  return COL.sports;
}
"""

code = re.sub(
    r"function getCSSCol\(name\).*?};", 
    new_col, 
    code,
    flags=re.DOTALL
)

# Add updateColors() to initTheme and toggleTheme
code = code.replace("initTheme();", "initTheme(); updateColors();")
code = code.replace("if (typeof gData !== 'undefined' && gData && gData.length > 0) drawAll();", "updateColors();")

with open('python_app/static/index.html', 'w') as f:
    f.write(code)

print("Performance fixed successfully.")
