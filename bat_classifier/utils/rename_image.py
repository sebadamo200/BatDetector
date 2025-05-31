import os

def rename_files_in_folder(folder_path):
    # Verify that the given path exists and is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Create a prefix by replacing path separators and spaces with underscores
    folder_name = folder_path.strip(os.sep).replace(os.sep, "_").replace(" ", "_")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            # Prepend the sanitized folder name to the original filename
            new_filename = f"{folder_name}_{filename}"
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file on disk
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} → {new_filename}")

# Prompt the user to enter a folder path and start renaming
folder_path = input("Enter the folder path: ").strip()
rename_files_in_folder(folder_path)

# run this script as python bat_classifier/utils/rename_image.py
# Then enter the folder path you want to rename files in.
# Example: C:\Users\YourName\Pictures\Bats.
