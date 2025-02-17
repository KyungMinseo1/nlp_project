from .nbq import nlp_before_qg

class Processor:
    def __init__(self, df):
        self.df = df
        self.nbq = None

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
            new_df = nbq.merge_to_df()
        else:
            new_df = self.nbq.merge_to_df()
        return new_df.to_json(orient='records', force_ascii=False, indent=4)
    
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
            new_df = nbq.merge_to_df()
        else:
            new_df = self.nbq.merge_to_df()
        return new_df.to_csv(encoding = "utf-8")