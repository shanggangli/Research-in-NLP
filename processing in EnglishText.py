import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def sentence_to_word(data):
    ps = PorterStemmer()
    letter = re.sub('https?://\S+|www\.\S+', '', data)  #去除网址
    words= re.sub(r'[^A-Za-z]+', ' ', letter).lower().split() # 清除不是字母的字符
    stop = set(stopwords.words('english'))
    clear_text = [ps.stem(w) for w in words if not w in stop]
    if len(clear_text)>2:
        return clear_text
