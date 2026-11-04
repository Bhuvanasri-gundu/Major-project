from pathlib import Path

target_file = Path(r"E:\Major_Project_Implementation\src\phase2\nli_validation.py")
lines = target_file.read_text(encoding="utf-8").splitlines()

# Filter out future and nullcontext from anywhere in first 20 lines
new_lines = []
for line in lines:
    if line.strip().startswith("from __future__ import annotations") or line.strip() == "from contextlib import nullcontext":
        continue
    new_lines.append(line)

final_text = "from __future__ import annotations\nfrom contextlib import nullcontext\n" + "\n".join(new_lines) + "\n"
target_file.write_text(final_text, encoding="utf-8")
print("Fixed imports in nli_validation.py successfully!")
