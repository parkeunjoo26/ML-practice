import pandas as pd
import re
from konlpy.tag import Okt

train_df=pd.read_csv('C:/anaconda/mydata/3/ratings_train.txt',sep='\t', encoding='utf-8')  #'cp949')
train_df.head(13)
train_df['label'].value_counts()

train_df=train_df.fillna(' ')
train_df['document']=train_df['document'].apply(lambda x:re.sub(r"\d+"," ",x))
test_df=pd.read_csv('C:/anaconda/mydata/3/ratings_test.txt',sep='\t', encoding='utf-8')  #'cp949')

test_df=test_df.fillna(' ')
test_df['document']=test_df['document'].apply(lambda x:re.sub(r"\d+"," ",x))
test_df.head(13)
train_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)
PYDEVD_DISABLE_FILE_VALIDATION=1

#PYDEVD_DISABLE_FILE_VALIDATION=1 
twitter=Okt()
def tw_tokenizer(text):
    tokens_ko=twitter.morphs(text)
    return tokens_ko
