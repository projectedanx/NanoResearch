with open("nanoresearch/agents/writing/latex_assembler.py", "r") as f:
    text = f.read()

# Fix \tabcolsep -> \\tabcolsep
text = text.replace("Auto-fix table overflow (inject \\small / \\tabcolsep / @{})", "Auto-fix table overflow (inject \\\\small / \\\\tabcolsep / @{})")

with open("nanoresearch/agents/writing/latex_assembler.py", "w") as f:
    f.write(text)
