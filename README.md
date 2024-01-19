## instruction_dataset 가공하기

### 대상 데이터셋
대상 데이터셋 : 한글 instruction dataset

### 목적
목적 : 여러가지 데이터셋 형식에서 사용하기 쉬운 instruction dataset을 일정 포맷(instruction, output, input)으로 변경하는 튜토리얼

### 변경 타입 1)
instruction, input, output으로 명확하게 column들이 주어져 있는 형식 : 이 형식의 경우 데이터의 column이 3개 이상이고 instruction, input, output의 형태가 아니더라도 유사한 column명이 주어지게 된다.

![image](https://github.com/LEE-hyeon0771/instruction_dataset/assets/84756586/7ddc7c9d-151a-4f8e-8000-47f536965fe1)

```
# instruction, input, output과 같은 column으로 이미 셋팅된 data
if len(df.columns) >= 3:
    df = df[args.args]
    df.columns = ['instruction', 'output', 'input']
```

위 방식으로 단순하게 instruction, output, input 순서로 인자를 나열해주고, 나머지 column 들은 버리는 방식을 선택한다.

- 사용방식

![image](https://github.com/LEE-hyeon0771/instruction_dataset/assets/84756586/b5df0100-c714-4a7a-8ce8-83cca27d63db)
```
python "input파일주소" [첫번째 column명] [두번째 column명] [세번째 column명] "output파일주소" 

이렇게 기록하게 되면 모든 내용들이 instruction, output, input의 순서로 정렬된 데이터프레임 형태의 json 파일로 변경 되게 된다.
```

### 변경 타입 2)
instruction, input, output으로 명확하게 column들이 주어져 있지 않는 방식

1. key-value 쌍이 이미 존재하는 경우(from, value) -> (ex) 'from' : 'human', 'from' : 'gpt')

![image](https://github.com/LEE-hyeon0771/instruction_dataset/assets/84756586/20f3f602-5866-493f-a090-5652ceb48a63)

```
target_column, first_delimiter, second_delimiter = args.args[0], args.args[1], args.args[2]

    instructions = []
    outputs = []
    
    for _, row in df.iterrows():
        conversation = row[target_column]
        human_msg = np.nan
        gpt_msg = np.nan

        # 1) key-value 쌍이 이미 존재하는 경우(from, value) -> (ex) 'from' : 'human', 'from' : 'gpt')
        if isinstance(conversation, list):
            for message in conversation:
                if message.get('from') == first_delimiter:
                    human_msg = message.get('value')
                elif message.get('from') == second_delimiter:
                    gpt_msg = message.get('value')
```
위 방식으로 먼저 타겟 column명을 선택하고, 첫 번째 구분자로 from : key를 가지는 human을 두 번째 구분자로 from : key를 가지는 gpt를 담게 되고, 그 값으로 value를 받게 된다.

- 사용방식

![image](https://github.com/LEE-hyeon0771/instruction_dataset/assets/84756586/ad427e9a-8db1-4285-9cc5-b7c70be721bb)

```
python "input파일주소" [target column명] [첫번째 구분자 human] [두번째 구분자 gpt] "output파일주소"

이렇게 기록하게 되면 모든 내용들이 instruction, output, input의 순서로 정렬된 데이터프레임 형태의 json 파일로 변경 되게 된다.
human의 내용은 instruction, gpt의 내용은 output, input은 결측값을 입력한다.
```

2. key-value 쌍이 없고, 단순히 구분자로 구분되어야 하는 경우 (ex) ### Human ~~~ ### Assistant)

![image](https://github.com/LEE-hyeon0771/instruction_dataset/assets/84756586/a0e43926-55dd-4f40-8cae-be2db032a19d)

```
# 2) key-value 쌍이 없고, 단순히 구분자로 구분되어야 하는 경우 (ex) ### Human ~~~ ### Assistant)
        elif isinstance(conversation, str):
            split_text = re.split(re.escape(first_delimiter) + '|' + re.escape(second_delimiter), conversation)
            if len(split_text) >= 3:
                human_msg = split_text[1].strip()
                gpt_msg = split_text[2].strip()
```
타겟 column명을 선택하고, 첫 번째 구분자로 "### Human", 두 번째 구분자로 "### Assistant"를 입력받아서 구분자들을 각 column으로 만들고, 내용들을 담게 된다.

- 사용방식

![image](https://github.com/LEE-hyeon0771/instruction_dataset/assets/84756586/9288a716-fd2a-4d82-89b4-c52cf999a0e6)

```
python "input파일주소" [target column명] [첫번째 구분자 ### Human] [두번째 구분자 ### Assistant] "output파일주소"

이렇게 기록하게 되면 모든 내용들이 instruction, output, input의 순서로 정렬된 데이터프레임 형태의 json 파일로 변경 되게 된다.
### Human의 내용은 instruction, ### Assistant의 내용은 output, input은 결측값을 입력한다.
```

### 데이터 통합

- 수 많은 데이터들을 위와 같은 방식으로 instruction, output, input 포맷으로 변경했다면 이제 데이터를 하나로 통합시켜야한다.

```
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
```

데이터를 통합시키기 위해 pandas의 concat 함수를 사용하고, json_file_paths 리스트에 위와 같은 방식으로 변경시킨 모든 json 파일들을 리스트로 담아주면, 한 번에 데이터를 통합시킬 수 있다.




