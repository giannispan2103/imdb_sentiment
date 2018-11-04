import re


def clean_en(string, lower=False):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower() if lower else string.strip()


class Review(object):
    """
    this class encapsulates data taken from imdb dataset
    data downloaded from http://ai.stanford.edu/~amaas/data/sentiment/
    """
    def __init__(self, text, summary, movie_id, init_score, sentiment):
        self.movie_id = movie_id
        self.init_score = init_score
        self.label = init_score / 10.0
        self.text = text
        self.cleaned_text = clean_en(self.text)
        self.tokens = self.cleaned_text.split()
        self.text_size = self.get_text_size()
        self.summary = summary
        self.cleaned_summary = clean_en(self.summary, False)
        self.summary_tokens = self.cleaned_summary.split()
        self.sentiment = sentiment

    def get_text_size(self):
        return len(self.tokens)

    def get_summary_size(self):
        return len(self.summary_tokens)

    def __eq__(self, other):
        return self.text == other.text

    def __ne__(self, other):
        return self.text != other.text

    def __hash__(self):
        return hash(self.text)



