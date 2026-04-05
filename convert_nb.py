import json
import ntpath

notebook_path = r"d:\PROject\Connect.ipynb"
script_path = r"d:\PROject\Connect.py"

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
    code = ""
    for idx, cell in enumerate(code_cells):
        code += f"# ------------ CELL {idx+1} ------------\n"
        code += "".join(cell.get("source", [])) + "\n\n"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Successfully converted to {script_path}")
except Exception as e:
    print(f"Error: {e}")
