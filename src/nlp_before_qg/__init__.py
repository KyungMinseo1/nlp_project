from .nbq import nlp_before_qg
import os
import csv
import json

class Processor:
    def __init__(self, df):
        self.df = df
        self.nbq = None
        
    def file_maker(self, path, data, format):
        if os.path.exists(path):
            user_input = input(f"File {path} already exists. Overwrite? (y/n): ").lower()
            if user_input != 'y':
                print(f"Skipping {path}")
                return
        if format == "json":        
            data.to_json(path, orient='records', force_ascii=False, indent=4)
            print(f"Final JSON results saved to: {path}")
        if format == "csv":
            data.to_csv(path, encoding = 'utf-8')
            print(f"Final CSV results saved to: {path}")

    def process_json(self):
        '''
        입력 받은 데이터프레임을 처리 후 json 파일 형식으로 반환
        '''
        if not self.nbq:
            nbq = nlp_before_qg(self.df)
            nbq.raw_text_extract()
            nbq.text_resub()
            nbq.split_sentence()
            nbq.summary_sentence()
            nbq.context_split()
            nbq.keybert()
            self.nbq = nbq
            new_df = nbq.merge_to_df()
        else:
            new_df = self.nbq.merge_to_df()
        final_path_json = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'final.json')    
        self.file_maker(final_path_json, new_df, 'json')
        
    
    def process_csv(self):
        '''
        입력 받은 데이터프레임을 처리 후 csv 파일 형식으로 반환
        '''
        if not self.nbq:
            nbq = nlp_before_qg(self.df)
            nbq.raw_text_extract()
            nbq.text_resub()
            nbq.split_sentence()
            nbq.summary_sentence()
            nbq.context_split()
            nbq.keybert()
            self.nbq = nbq
            new_df = nbq.merge_to_df()
        else:
            new_df = self.nbq.merge_to_df()
        final_path_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'final.csv')
        self.file_maker(final_path_csv, new_df, 'csv')