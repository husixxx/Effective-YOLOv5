import os

folder_path = "/home/husix/BP_2024/data/labels/val"

empty_files = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
        empty_files.append(filename)

if empty_files:
    print("Prázdné soubory:")
    for empty_file in empty_files:
        print(empty_file)
else:
    print("Žádné prázdné soubory nebyly nalezeny.")

