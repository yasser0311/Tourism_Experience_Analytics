import nbformat
from nbconvert import MarkdownExporter

def generate_docs():
    notebooks = ["main.ipynb", "rating_prediction.ipynb", "user_visit_prediction.ipynb",
                 "recommendation_suggestion.ipynb",  "EDA.ipynb"] # Add all notebooks
    
    for nb_file in notebooks:
        # Convert notebook to markdown
        with open(nb_file) as f:
            nb = nbformat.read(f, as_version=4)
        
        md, _ = MarkdownExporter().from_notebook_node(nb)
        
        # Save with consistent naming
        with open(f"docs/notebooks/{nb_file.replace('.ipynb','.md')}", "w") as f:
            f.write(md)

if __name__ == "__main__":
    generate_docs()