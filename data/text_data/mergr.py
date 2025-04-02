import pandas as pd

# # Read the two Excel files
# df1 = pd.read_csv("C:/Users/drv/drv_codes/dl_project/text/modify_fake.csv", nrows = 23503, usecols=[0, 1, 2, 3, 4])  # Replace with your actual filename
# df2 = pd.read_csv("C:/Users/drv/drv_codes/dl_project/text/True.csv", nrows = 21418, usecols=[0, 1, 2, 3, 4])  # Replace with your actual filename

# # Merge (concatenate) dataframes row-wise
# df_merged = pd.concat([df1, df2], ignore_index=True)

# # Save the merged dataframe to a new Excel file
# df_merged.to_csv("merged_files.csv", index=False)

# print("Merged file saved as 'merged_files.csv'.")

df = pd.read_csv("merged_files.csv")

# Shuffle the rows
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled file
df_shuffled.to_csv("shuffled_merged.csv", index=False)

print("Shuffled the merged data and saved as 'shuffled_merged.csv'.")