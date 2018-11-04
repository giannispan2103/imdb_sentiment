from googletrans import Translator
import time
import pandas as pd

from preprocess import get_posts


def translate(translator, text, src="en", dest="de"):
    # translation = Translator()
    try:
        translation = translator.translate(text, src=src, dest=dest)
        return translation.text
    except:
        time.sleep(10)
        print('first catch:::', src, "->", dest)
        try:
            translation = translator.translate(text, src=src, dest=dest)
            return translation.text
        except:
            time.sleep(10)
            print("second catch:::",src, "->",  dest)
            try:
                print("third catch:::", src, "->", dest)
                translation = translator.translate(text, src=src, dest=dest)
                return translation.text
            except:
                print("return")
                return "$$"


def translate_back(translator, text):
    translation = translate(translator, text, "en", "de")
    if translation == "$$":
        return "$$"
    else:
        return translate(translator, translation, "de", "en")


def translate_reviews(data, start, end):
    translator = Translator()
    texts, summaries, movies, init_scores, sentiments = [], [], [], [], []
    for i, d in enumerate(data[start:end], start=1):
        print(i)
        text = translate_back(translator, d.text)
        if text != "$$":
            texts.append(text)
            summaries.append(d.summary)
            movies.append(d.movie_id)
            init_scores.append(d.init_score)
            sentiments.append(d.sentiment)

    df = pd.DataFrame({'translation': texts, 'summary': summaries, 'movie': movies,
                       'init_score': init_scores, 'sentiment': sentiments})
    df.to_csv("translations%d-%d.csv" % (start, end), index=False, encoding="utf8")


if __name__ == "__main__":
    data = get_posts("train")
    print(len(data))
    translate_reviews(data, 2800, 3000)
