import pandas as pd

json_file_paths = [
    'C:/Users/DEEPNOID/Desktop/1_format/real_file/real_combined_data2.json',
    'C:/Users/DEEPNOID/Desktop/1_format/real_file/real_combined_data.json',
]

dataframes = []

for path in json_file_paths:
    try:
        df = pd.read_json(path)
        dataframes.append(df)
        print(df.head())
    except Exception as e:
        print(f"Error reading {path}: {e}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)

    output_file_path = 'C:/Users/DEEPNOID/Desktop/1_format/real_file/Final_combine.json'
    combined_df.to_json(output_file_path, orient='records', force_ascii=False)
    print(f"Combined data saved to {output_file_path}")
else:
    print("No dataframes to combine.")

'''
import pandas as pd

json_file_path1 = 'C:/Users/DEEPNOID/Desktop/2_format/new_file2/combined_data.json'

json_df1 = pd.read_json(json_file_path1)

parquet_file_path1 = 'C:/Users/DEEPNOID/Desktop/2_format/train-00000-of-00001-4dbc07c8282bce17.parquet'
parquet_file_path2 = 'C:/Users/DEEPNOID/Desktop/2_format/train-00000-of-00001-f842f8e276141f79.parquet'
parquet_df1 = pd.read_parquet(parquet_file_path1)
parquet_df2 = pd.read_parquet(parquet_file_path2)

combined_df = pd.concat([json_df1, parquet_df1, parquet_df2], ignore_index=True)

combined_json = combined_df.to_json(orient='records', force_ascii=False)

output_file_path = 'C:/Users/DEEPNOID/Desktop/2_format/new_file2/combined_data2.json'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(combined_json)'''