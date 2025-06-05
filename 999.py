import pathlib

# 4→4 對應： ' → '， ' → '， " → "， " → "
sub = str.maketrans("''""", "''\"\"")

for p in pathlib.Path('.').rglob('*.py'):
    txt = p.read_text(encoding='utf-8')
    new = txt.translate(sub)
    if txt != new:
        p.write_text(new, encoding='utf-8')
        print(f"fixed {p}")
