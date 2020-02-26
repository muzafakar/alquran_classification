class Preprocessor:

    def __init__(self, dataframe):
        self._dataframe = dataframe

    def __clean(self, txt: str):
        """case-folding"""
        print('case-folding...')
        txt = txt.lower()
        """using default puntuation list from python"""
        from string import punctuation
        """modifying punctuatioin list for quranic verses needs"""
        print('removing punctuation...')
        txt = txt.translate(txt.maketrans('', '', punctuation))
        return txt

    def __filter(self, txt: str):
        """filtering words using indonesian nltk punkt stopwords list"""
        print('removing stopwords...')
        from nltk.corpus import stopwords
        stop_word = set(stopwords.words('indonesian'))
        words = txt.split(' ')
        cleaned = [word for word in words if word not in stop_word]
        return ' '.join(cleaned)

    def __stem(self, txt: str):
        """stemming words using sastrawi (Nazief Andriani Algorithm)"""
        print('stemming...')
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stm = factory.create_stemmer()
        result = stm.stem(txt)
        print(result)
        return result

    def __tokenize(self, txt: str):
        """tokenizing words"""
        print('tokenizing...')
        result = [word for word in txt.split(' ')]
        return result

    def __generate_bow(self, tokens: list):
        """feature selection using Bag of Words representation"""
        print('creating bow...')
        bow = {}
        for token in tokens:
            bow[token] = bow[token] + 1 if token in bow else 1
        return bow

    def preprocess(self, txt: str):
        cleaned = self.__clean(txt)
        filtered = self.__filter(cleaned)
        stemmed = self.__stem(filtered)
        tokenized = self.__tokenize(stemmed)
        bow = self.__generate_bow(tokenized)
        return tokenized


import pandas as pd

if __name__ == '__main__':
    header = ['chapter', 'verse', 'text']
    header2 = ['chapter', 'verse', 'text', 'prep']
    filename = "../data/quran_unlabelled.csv"
    df = pd.read_csv(filename)
    prep = Preprocessor(df)

    new_data = []
    for verse in df.itertuples():
        row = {}
        for key in header:
            row[key] = getattr(verse, key)
            if key == 'text':
                row['prep'] = " ".join(prep.preprocess(row[key]))
        new_data.append(row)

    filename = "quran_unlabelled_prep.csv"
    data = pd.DataFrame(new_data)
    data.to_csv(filename, header=header2, index=False)
