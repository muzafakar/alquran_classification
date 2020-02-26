from data.quran import Quran
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold

filename = "../data/quran_labelled.csv"
quran = Quran(filename)
# stat.multi_labelled_verse_count(True)
# stat.count_verse_by_topic(True)


# verse = 'Allah tidak membebani seseorang melainkan sesuai dengan kesanggupannya. Ia mendapat pahala (dari ' \
#         'kebajikan) yang diusahakannya dan ia mendapat siksa (dari kejahatan) yang dikerjakannya. (Mereka ' \
#         'berdoa): "Ya Tuhan kami, janganlah Engkau hukum kami jika kami lupa atau kami tersalah. Ya Tuhan kami, ' \
#         'janganlah Engkau bebankan kepada kami beban yang berat sebagaimana Engkau bebankan kepada orang-orang ' \
#         'sebelum kami. Ya Tuhan kami, janganlah Engkau pikulkan kepada kami apa yang tak sanggup kami memikulnya. ' \
#         'Beri maaflah kami; ampunilah kami; dan rahmatilah kami. Engkaulah Penolong kami, maka tolonglah kami ' \
#         'terhadap kaum yang kafir". '


X, Y = quran.get_preprocessed()
print(X)
