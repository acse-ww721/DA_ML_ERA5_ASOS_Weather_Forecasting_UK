# Name: Wenqi Wang
# Github username: acse-ww721

import os


def get_current_directory():
    """
    Get the absolute path of the directory where the current script or Jupyter Notebook is running.

    Returns:
        str: The absolute path of the current directory.

    Raises:
        OSError: If it's unable to determine the absolute path of the current directory.

    Example:
        >>> get_current_directory()
        '/path/to/your/current/directory'
    """
    if "__file__" in globals():
        # Running in a Python file
        return os.path.abspath(os.path.dirname(__file__))
    else:
        # Running in a Jupyter Notebook
        return os.path.abspath(os.path.dirname(""))


import os


def find_folder(c, data_folder, data_category, output_folder):
    """
    Find the path to a specific folder based on input parameters.

    Args:
        c (str): The country code or identifier.
        data_folder (str): The main data folder.
        data_category (str): The category of data within the data folder.
        output_folder (str): The output folder name.

    Returns:
        str: The absolute path to the specified folder.

    Example:
        >>> find_folder("GB", "data", "raw_data", "ASOS_DATA")
        '/path/to/your/project/data/sales/output/US_output'
    """
    current_directory = get_current_directory()
    project_root = os.path.abspath(os.path.join(current_directory, "..", "."))
    folder_name = f"{c}_{output_folder}"
    folder_path = os.path.join(
        project_root, data_folder, data_category, output_folder, folder_name
    )

    return folder_path


def create_folder(c, data_folder, data_category, output_folder):
    """
    Create a folder with a specific name based on input parameters.

    Args:
        c (str): The country code or identifier.
        data_folder (str): The main data folder.
        data_category (str): The category of data within the data folder.
        output_folder (str): The output folder name.

    Returns:
        str: The absolute path to the created folder.

    Example:
        >>> create_folder("GB", "data", "raw_data", "ASOS_DATA")
        'path/to/your/project/data/sales/output/US_output'
    """
    folder_path = find_folder(c, data_folder, data_category, output_folder)

    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")

    return folder_path


# Test example
# project_root = "/path/to/project"
# country = ["WWQ"]
# data_folder = "data"
# data_category = "raw_data"
# output_folder = "ASOS_DATA"
# create_folder(country[0],data_folder, data_category, output_folder)
