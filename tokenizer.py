import json
from transformers import AutoTokenizer

MODEL = "42dot/42dot_LLM-SFT-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_text(text):
    if text is None:
        return 0
    return len(tokenizer.tokenize(text))

def process_file(file_path):
    
    total_instruction_tokens = 0
    total_output_tokens = 0
    total_input_tokens = 0
    total_tokens = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            i = json.loads(line)
        
            instruction_tokens = tokenize_text(i['instruction'])
            output_tokens = tokenize_text(i['output'])
            input_tokens = tokenize_text(i['input'])
            
            total_instruction_tokens += instruction_tokens
            total_output_tokens += output_tokens
            total_input_tokens += input_tokens
            
            print(f"Instruction Tokens: {instruction_tokens}, Output Tokens: {output_tokens}, Input Tokens: {input_tokens}")
    total_tokens = total_instruction_tokens + total_output_tokens + total_input_tokens
    print(f"Total Instruction Tokens: {total_instruction_tokens}, Total Output Tokens: {total_output_tokens}, Total Input Tokens: {total_input_tokens}, Total Tokens: {total_tokens}")

json_file_path = 'C:/Users/DEEPNOID/Desktop/instruction, output, input type/preprocessing file/Final_combine.json'
process_file(json_file_path)