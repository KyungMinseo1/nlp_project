from src.nlp_before_qg import Processor
import os
import json
import pandas as pd

def main():
    input_data = os.path.join(os.path.dirname(__file__), 'data', 'result.json')
    input = pd.read_json(input_data)
    
    nlp = Processor(input)
    data_json = nlp.process_json()
    data_csv = nlp.process_csv()
    
    final_path_json = os.path.join(os.path.dirname(__file__), 'output', 'final.json')
    final_path_csv = os.path.join(os.path.dirname(__file__), 'output', 'final.csv')
    
    def write_file(path, data, write_function):
        if os.path.exists(path):
            user_input = input(f"File {path} already exists. Overwrite? (y/n): ").lower()
            if user_input != 'y':
                print(f"Skipping {path}")
                return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            write_function(f, data)
        print(f"File saved: {path}")
        
    # Write JSON file
    write_file(final_path_json, data_json, lambda f, data: json.dump(data, f, ensure_ascii=False, indent=4))
    
    # Write CSV file
    write_file(final_path_csv, data_csv, lambda f, data: csv.writer(f).writerows(data))

if __name__ == "__main__":
    main()