import pandas as pd
import os

# Path to the folder containing CSV files
folder_path = "../data/audio/"


# Function to extract text column from CSV and save to txt file
def csv_to_txt(csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file)
    # Group DataFrame by the 'dominant' column
    grouped = df.groupby('dominant')

    # Extracting file name from path
    file_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Iterate over groups
    for group_name, group_data in grouped:
        # Saving group data to txt file
        txt_file_path = os.path.join(folder_path + "txt", f"{file_name}_{group_name}.txt")
        with open(txt_file_path, 'w') as txt_file:
            for text in group_data['text']:
                txt_file.write(str(text) + ' ')


# Iterate over CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        csv_file_path = os.path.join(folder_path, file_name)
        csv_to_txt(csv_file_path)
