import os


def get_current_directory():
    if "__file__" in globals():
        # Running in a Python file
        return os.path.abspath(os.path.dirname(__file__))
    else:
        # Running in a Jupyter Notebook
        return os.path.abspath(os.path.dirname(""))


def find_folder(c, data_folder, data_category, output_folder):
    # c: country list
    current_directory = get_current_directory()
    project_root = os.path.abspath(os.path.join(current_directory, "..", "."))
    folder_name = f"{c}_{output_folder}"
    folder_path = os.path.join(
        project_root, data_folder, data_category, output_folder, folder_name
    )

    return folder_path


def create_folder(c, data_folder, data_category, output_folder):
    # c: country list
    folder_path = find_folder(c, data_folder, data_category, output_folder)

    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")

    return folder_path

    # ... rest of the function ...


# Test example
# project_root = "/path/to/project"
country = ["WWQ"]
data_folder = "data"
data_category = "raw_data"
output_folder = "ASOS_DATA"
# create_folder(country[0],data_folder, data_category, output_folder)
