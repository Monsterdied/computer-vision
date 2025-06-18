import nbformat as nbf
def merge_notebooks(filenames):
    merged = nbf.v4.new_notebook()
    for fname in filenames:
        with open(fname, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
            merged.cells.extend(nb.cells)
    return merged
notebooks_to_merge = ["./differentYolos.ipynb", "./task3_2_ResNeXt101_64_x4DLabel2_DataAugmentation.ipynb","./task3_2_Tomás.ipynb"] # Adjust paths as needed
merged_notebook = merge_notebooks(notebooks_to_merge)
with open("merged_output.ipynb", "w", encoding='utf-8') as f:
    nbf.write(merged_notebook, f)