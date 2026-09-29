# nl2logic README demo. From the repo root:
#   docker build -t nl2logic-vhs docs/demo
#   docker run --rm -v "$PWD:/repo" -v "$PWD/docs/demo:/vhs" nl2logic-vhs demo.tape
Output demo.gif


Require python

Set Shell "bash"
Set FontSize 19
Set Width 1400
Set Height 820
Set Padding 32
Set LineHeight 1.25
Set TypingSpeed 38ms
Set Theme { "name": "maganacasa", "background": "#0b0b0b", "foreground": "#f2f0ea", "cursor": "#e2231a", "selection": "#333333", "black": "#0b0b0b", "red": "#e2231a", "green": "#7bd88f", "yellow": "#e6c07b", "blue": "#7aa2f7", "magenta": "#c678dd", "cyan": "#56b6c2", "white": "#f2f0ea", "brightBlack": "#6b6b6b", "brightRed": "#ff5c50", "brightGreen": "#9ee6a8", "brightYellow": "#f0d399", "brightBlue": "#9bb8ff", "brightMagenta": "#d7a1e7", "brightCyan": "#7fd0da", "brightWhite": "#ffffff" }

Env LANG "C.UTF-8"
Env LC_ALL "C.UTF-8"

Hide
Type `cd /repo && export PS1='\[\e[38;2;226;35;26m\]nl2logic\[\e[0m\] $ '`
Enter
Type `ask(){ python scripts/reason_ontology.py --accepted data/benchmarks/fm2-0/seed_positive.jsonl --background examples/reasoning/background.jsonl --query "$1" | sed -e '/^Proof and evidence/d' -e 's/^PROVEN/\x1b[1;32mPROVEN\x1b[0m/' -e 's/^UNKNOWN/\x1b[1;33mUNKNOWN\x1b[0m/' | fold -s -w 104; }`
Enter
Type "clear"
Enter
Show

Sleep 600ms
Type "# Reviewed FM 2-0 doctrine, formalized to KIF and checked by the Vampire prover"
Enter
Sleep 900ms
Type "# Is human intelligence a kind of procedure?"
Enter
Type "ask '(subclass HumanIntelligence Procedure)'"
Sleep 400ms
Enter
Sleep 6s

Type "# Is it a warfighting function? The doctrine never says so."
Enter
Type "ask '(subclass HumanIntelligence WarfightingFunction)'"
Sleep 400ms
Enter
Sleep 4.5s
