from src.nlp_before_qg import Processor
import os
import pandas as pd

def main():
    input_data = os.path.join(os.path.dirname(__file__), 'data', 'result.json')
    input_df = pd.read_json(input_data)
    
    nlp = Processor(input_df)

    data_format = str(input("원하는 파일 형식 지정(json/csv/both)")).lower()

    if data_format == "json":
        nlp.process_json()
    elif data_format == "csv":
        nlp.process_csv()
    else:
        nlp.process_json()
        nlp.process_csv() 

if __name__ == "__main__":
    main()