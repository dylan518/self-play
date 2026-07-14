import ast, sys

p = sys.argv[1]
src = open(p).read()
a = """    results = []
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)"""
r = """    results = []
    # (removed upstream debug dump of predicts to test.json in CWD — it wrote to $HOME
    # on every reward call and killed both unison runs when the 30GB quota filled)"""
if a in src:
    src = src.replace(a, r, 1)
    ast.parse(src)
    open(p, "w").write(src)
    print("FIXED", p)
else:
    print("anchor absent (already fixed?)", p)
