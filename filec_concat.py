import pandas as pd

json_file_paths = [

    "C:/Users/DEEPNOID/Desktop/human, bot type/preprocessing file/Final_combine2.json",
    "C:/Users/DEEPNOID/Desktop/instruction, output, input type/preprocessing file/Final_combine1.json"
]

dataframes = []

for path in json_file_paths:
    try:
        df = pd.read_json(path, lines=True)
        dataframes.append(df)
        #print(df.head())
    except Exception as e:
        print(f"Error reading {path}: {e}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)

    output_file_path = 'C:/Users/DEEPNOID/Desktop/instruction, output, input type/preprocessing file/Final_combine.json'
    combined_df.to_json(output_file_path, orient='records', force_ascii=False, lines=True)
    print(f"Combined data saved to {output_file_path}")
else:
    print("No dataframes to combine.")