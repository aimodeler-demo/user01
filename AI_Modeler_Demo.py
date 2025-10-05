"""
Source: AI_Modeler_Demo.ipynb
Version: 1.0.0
Exported: 2025-10-01 21:35:22
Python: 3.12.3
"""


!docker push ajbrownv/ai_modeler_demo:latest

import nbformat

input_path = "AI_Modeler_Demo.ipynb"      

cell_indexes = [1, 2]                     

output_path = "AI_Modeler_Demo.py"         

def export_notebook():

    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    code_snippets = []
    for i in cell_indexes:
        if i < len(nb.cells):
            cell = nb.cells[i]
            if cell.cell_type == 'code':
                code_snippets.append(cell['source'])
            else:
                print(f"Cell {i} is not a code cell. Skipped.")
        else:
            print(f"Cell {i} is out of range. Skipped.")

    code = "\n\n".join(code_snippets)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)

    print(f"Done! Code saved to {output_path}")

convert_notebook()