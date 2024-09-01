import pandas as pd

# Step 1: Read the CSV file
csv_file_path = 'bus_maintenance_data.csv'  # Replace with your actual CSV file path
try:
    df = pd.read_csv(csv_file_path)
    print("CSV file read successfully.")
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    exit()

# Step 2: Save the DataFrame as a PKL file
pkl_file_path = 'bus_maintenance_data.pkl'  # Replace with your desired PKL file path
try:
    df.to_pickle(pkl_file_path)
    print(f"CSV file has been converted to PKL and saved as {pkl_file_path}")
except Exception as e:
    print(f"Error: {e}")