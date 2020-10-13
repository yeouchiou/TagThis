from collections import defaultdict
import pandas as pd


class Articles():
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self._preprocessToDF()

    def _preprocessToDF(self, urls=True):
        texts = []
        titles = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            url = f.readline()
            while True:
                if not url:
                    break
                if url[:3] == 'URL':
                    if not urls:
                        titles.append(url.strip().split('/')[-1].split('.')[0].replace('-', ' '))
                    else:
                        # Approximate titles by url
                        titles.append(url.strip().split()[-1])
                    line = f.readline()
                    currtext = ''
                    while line and line[:3] != 'URL':
                        if line == '\n':
                            line = f.readline()
                            continue
                        currtext += line.replace('\n', ' ')
                        line = f.readline()

                    texts.append(currtext.strip())
                    url = line
        dd = defaultdict(list)
        for i in range(len(titles)):
            dd['title'].append(titles[i])
            dd['text'].append(texts[i])
        df = pd.DataFrame(dd)
        # drop rows with no text
        # Need to alter this if using a different dataset
        df.drop([3978, 7096, 7108, 8869], inplace=True)
        return df
