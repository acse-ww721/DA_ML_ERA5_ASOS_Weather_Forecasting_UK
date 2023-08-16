import os


def generate_tree(directory, indent="", output_file=None, skip_folders=None):
    line = indent + "|-- " + os.path.basename(directory) + "\n"
    if os.path.isdir(directory):
        items = sorted(os.listdir(directory))
        for item in items:
            item_path = os.path.join(directory, item)
            if skip_folders and os.path.basename(item_path) in skip_folders:
                continue  # Skip specified folders
            if os.path.isdir(item_path):
                line += generate_tree(
                    item_path, indent + "    ", skip_folders=skip_folders
                )
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

    # List of folders to skip
    folders_to_skip = [
        ".git",
        ".DS_Store",
        "logs",
        "objects",
        "refs",
        ".idea",
        ".ipynb_checkpoints",
        "script_",
    ]

    generate_tree(
        root_directory, output_file=output_file_path, skip_folders=folders_to_skip
    )
    print("File tree structure generated and saved to 'tree_structure.txt'")
