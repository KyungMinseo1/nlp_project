from src.nlp_before_qg import Processor
import os
import pandas as pd

def main():

    input_dir = os.path.join(os.path.dirname(__file__), 'data', 'source_data')
    with os.scandir(input_dir) as entries:
        json_files = [entry.name for entry in entries if entry.is_file() and entry.name.endswith('.json')]
    
    if not json_files:
        print("해당 디렉토리에 폴더가 없습니다.")
        return None

    print("사용 가능한 json 폴더 목록:")
    for idx, folder in enumerate(json_files, 1):
        print(f"{idx}. {folder}")

    while True:
        try:
            choice = int(input("사용할 폴더 번호를 입력하세요: ")) - 1
            if 0 <= choice < len(json_files):
                selected_file = json_files[choice]
                break
            else:
                print("잘못된 번호입니다. 다시 입력해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

    selected_path = os.path.join(input_dir, selected_file)

    input_df = pd.read_json(selected_path)
    
    nlp = Processor(input_df)

    while True:
        try:
            data_format = str(input("원하는 파일 형식 지정(json/csv/both): ")).lower()
            if data_format == "json":
                nlp.process_json()
                break
            elif data_format == "csv":
                nlp.process_csv()
                break
            elif data_format == "both":
                nlp.process_json()
                nlp.process_csv()
                break
            else:
                print("잘못된 입력입니다. 다시 입력해주세요.")
        except ValueError:
            print("다시 입력해주세요.")

if __name__ == "__main__":
    main()