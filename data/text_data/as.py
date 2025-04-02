import pandas as pd

# Specify the input Parquet file and output CSV file paths
input_parquet = "C:\\Users\\drv\\Downloads\\yellow.parquet"  # Replace with your Parquet file path
output_csv = "output_file.csv"        # Replace with your desired CSV output path

# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet(input_parquet)

# Write the DataFrame to a CSV file (without the DataFrame index)
df.to_csv(output_csv, index=False)

print(f"Conversion complete. CSV file saved as {output_csv}")
