import pandas as pd


class Quran:

    def __init__(self, filename: str):
        self._df = pd.read_csv(filename)
        # self._prep = Preprocessor(dataframe)
        # self._count_vectorizer = CountVectorizer(dtype=np.int32)
        self._colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # to generate topics/label
    def get_topics(self):
        df_topic = self._df.drop(['chapter', 'verse', 'text'], axis=1)
        return list(df_topic.columns.values)

    def count_verse_by_topic(self, show_plot: bool = False):
        topics = self.get_topics()
        counts = []
        for t in topics:
            counts.append((t, self._df[t].sum()))
        import pandas as pd
        df_stat = pd.DataFrame(counts, columns=['topik', 'banyak ayat'])
        if show_plot:
            import matplotlib.pyplot as plt
            df_stat.plot.bar(x='topik', y='banyak ayat', legend=False, color=self._colors)
            # plt.title('Distribusi ayat berdasarkan topik')
            plt.ylabel('banyak ayat')
            plt.xlabel('topik')
            for t_index in range(0, len(topics)):
                y = df_stat['banyak ayat'].iloc[t_index]
                plt.text(t_index, y + 50, str(y), va='center', ha='center')
            plt.show()

        return df_stat

    def multi_labelled_verse_count(self, show_plot: bool = False):
        # menghitung jumlah label per-ayat
        rowsum = self._df.iloc[:, 3:].sum(axis=1)
        # menghitung jumlah ayat berdasarkan jumlah label
        result = rowsum.value_counts()
        if show_plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            axs = sns.barplot(result.index, result.values)
            # plt.title('Distribusi ayat berdasarkan banyak label')
            plt.ylabel('banyak ayat')
            plt.xlabel('ayat dengan jumlah n label')
            # showing value of each bar
            for x, y in enumerate(result):
                plt.text(x, y + 30, str(y), va='center', ha='center', fontweight='bold')

            plt.show()

        return result

    def verse_with_no_label(self):
        return self._df['text'].isnull().sum()

    def get_preprocessed(self):
        return "null"
