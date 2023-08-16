import os
from folder_utils import get_current_directory
def generate_tree(directory, indent="", output_file=None):
    line = indent + "|-- " + os.path.basename(directory) + "\n"
    if os.path.isdir(directory):
        items = sorted(os.listdir(directory))
        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                line += generate_tree(item_path, indent + "    ")
            else:
                line += indent + "    |-- " + item + "\n"
    if output_file:
        with open(output_file, "a") as f:
            f.write(line)
    return line


if __name__ == "__main__":
    root_directory = "../"  # Root directory of the file system
    output_file_path = "tree_structure.txt"

    with open(output_file_path, "w") as f:
        f.write("File Tree Structure:\n")

    generate_tree(root_directory, output_file=output_file_path)
    print("File tree structure generated and saved to 'tree_structure.txt'")
