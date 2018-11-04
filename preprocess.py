import numpy as np
from review import Review
import os
import random
import pandas as pd
from paths import TRANSLATIONS_PATH, DATA_PATH, SUMMARIES_PATH, EMBEDDINGS_PATH
PAD_TOKEN = "*$*PAD*$*"
UNK_TOKEN = "*$*UNK*$*"
UNK_MOV = "$#$#Nan#$#$"

VAL_DICT = {'pos': 1,
            'neg': 0}


def create_batches(reviews, w2i, m2i, pad_tnk=PAD_TOKEN, unk_tkn=UNK_TOKEN, unk_movie=UNK_MOV, batch_size=128,
                   max_len=100, summary_maxlen=100, sort_data=True):
    """
    :param reviews: a list of Review objects
    :param w2i: a word-to-index dictionary with all embedded words that will be used in training
    :param pad_tnk: the pad token
    :param unk_tkn: the unknown token
    :param unk_movie: the unknown movie
    :param batch_size: how many posts will be in every batch
    :param max_len: the padding size for the texts
    :param sort_data: boolean indicating if the list of posts  will be sorted by the size of the text
    :param m2i: a movie-to-index dictionary
    :param unk_movie: unk movie
    :return: a list of batches
    """
    if sort_data:
        reviews.sort(key=lambda x: -len(x.tokens))
    offset = 0
    batches = []
    while offset < len(reviews):
        batch_texts = []
        batch_movies = []
        batch_labels = []
        batch_sentiments = []
        batch_summaries = []
        start = offset
        end = min(offset + batch_size, len(reviews))
        for i in range(start, end):
            batch_max_size = reviews[start].text_size if sort_data else max(list(map(lambda x: x.text_size, reviews[start:end])))
            batch_summary_max_size = max(list(map(lambda x: x.text_size, reviews[start:end])))
            batch_texts.append(get_indexed_text(w2i, pad_text(reviews[i].tokens, max(min(max_len, batch_max_size), 1), pad_tkn=pad_tnk), unk_token=unk_tkn))
            batch_summaries.append(get_indexed_text(w2i, pad_text(reviews[i].summary_tokens, max(min(summary_maxlen, batch_summary_max_size), 1), pad_tkn=pad_tnk), unk_token=unk_tkn))
            batch_movies.append(get_indexed_value(m2i, reviews[i].movie_id, unk_movie))
            batch_labels.append(reviews[i].label)
            batch_sentiments.append(reviews[i].sentiment)

        batches.append({'text': np.array(batch_texts),
                        'summary': np.array(batch_summaries),
                        'movie': np.array(batch_movies),
                        'label': np.array(batch_labels, dtype='float32'),
                        'sentiment': np.array(batch_sentiments)})
        offset += batch_size
    return batches


def get_embeddings(path=EMBEDDINGS_PATH, size=50):
    """
    :param path: the directory where all glove twitter embeddings are stored.
    glove embeddings can be downloaded from https://nlp.stanford.edu/projects/glove/
    :param size: the size of the embeddings. Must be in [25, 50, 100, 200]
    :return: a word-to-list dictionary with the embedded words and their corresponding embedding
    """
    embeddings_dict = {}
    f_path = path % size
    with open(f_path,'r', encoding='utf8') as f:
        for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
    return embeddings_dict


def create_freq_vocabulary(tokenized_texts):
    """
    :param tokenized_texts: a list of lists of tokens
    :return: a word-to-integer dictionary with the value representing the frequency of the word in data
    """
    token_dict = {}
    for text in tokenized_texts:
        for token in text:
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
    return token_dict


def get_frequent_words(token_dict, min_freq):
    """
    :param token_dict: a word-to-integer dictionary with the value representing the frequency of the word in data
    :param min_freq: the minimum frequency
    :return: the list with tokens having frequency >= min_freq
    """
    return [x for x in token_dict if token_dict[x] >= min_freq]


def create_final_dictionary(reviews,
                            min_freq,
                            unk_token,
                            pad_token,
                            embeddings_dict):
    """
    :param reviews: a list of Post objects
    :param min_freq: the min times a word must be found in data in order not to be considered as unknown
    :param unk_token: the unknown token
    :param pad_token: the pad token
    :param embeddings_dict: a word-to-list dictionary with the embedded words and their corresponding embedding
    :return: a word-to-index dictionary with all the words that will be used in training
    """
    tokenized_texts = [x.tokens + x.summary_tokens for x in reviews] # + [x.summary_tokens for x in reviews]
    voc = create_freq_vocabulary(tokenized_texts)
    print("tokens found in training data set:", len(voc))
    freq_words = get_frequent_words(voc, min_freq)
    print("tokens with frequency >= %d: %d" % (min_freq, len(freq_words)))
    words = list(set(freq_words).intersection(embeddings_dict.keys()))
    print("embedded tokens with frequency >= %d: %d" % (min_freq,len(words)))
    words = [pad_token, unk_token] + words
    return {w: i for i, w in enumerate(words)}


def create_movie_dictionary(review, min_freq):
    movies = [x.movie_id for x in review]
    movie_dict = {}
    for movie in movies:
        try:
            movie_dict[movie] += 1
        except KeyError:
            movie_dict[movie] = 1
    final_movies = [UNK_MOV] + [x for x in movie_dict if movie_dict[x] >= min_freq]
    return {m: i for i, m in enumerate(final_movies)}


def get_embeddings_matrix(word_dict, embeddings_dict, size):
    """
    :param word_dict: a word-to-index dictionary with the tokens found in data
    :param embeddings_dict: a word-to-list dictionary with the embedded words and their corresponding embedding
    :param size: the size of the word embedding
    :return: a matrix with all the embeddings that will be used in training
    """
    embs = np.zeros(shape=(len(word_dict), size))
    for word in word_dict:
        try:
            embs[word_dict[word]] = embeddings_dict[word]
        except KeyError:
            print('no embedding for: ', word)
    embs[1] = np.mean(embs[2:])

    return embs


def get_indexed_value(w2i, word, unk_token):
    """
    return the index of a token in a word-to-index dictionary
    :param w2i: the word-to-index dictionary
    :param word: the token
    :param unk_token: to unknown token
    :return: an integer
    """
    try:
        return w2i[word]
    except KeyError:
        return w2i[unk_token]


def get_indexed_text(w2i, words, unk_token):
    """
    return the indices of the all the tokens in a list in a word-to-index dictionary
    :param w2i: the word-to-index dictionary
    :param words: a list of tokens
    :param unk_token: to unknown token
    :return: a list of integers
    """
    return [get_indexed_value(w2i, word, unk_token) for word in words]


def pad_text(tokenized_text, maxlen, pad_tkn):
    """
    fills a list of tokens with pad tokens if the length of the list is larger than maxlen
    or return the maxlen last tokens of the list
    :param tokenized_text: a list of tokens
    :param maxlen: the max length
    :param pad_tkn: the pad token
    :return: a list of tokens
    """
    if len(tokenized_text) < maxlen:
        return [pad_tkn] * (maxlen - len(tokenized_text)) + tokenized_text
    else:
        return tokenized_text[len(tokenized_text) - maxlen:]


def load_data_per_category(path_dir, category, summary_dict):
    data = []
    data_dir = DATA_PATH + path_dir + "\\" + category + "\\"
    urls_path = DATA_PATH + path_dir + "\\urls_%s.txt" % category
    with open(urls_path, 'r', encoding='utf8') as f:
        urls = f.readlines()
    paths = os.listdir(data_dir)
    for path in paths:
        doc = int(path.split(".")[0].split("_")[0])
        score = int(path.split(".")[0].split("_")[1])
        with open(data_dir + path, 'r', encoding='utf8') as f:
            text = f.read()
        movie_id = urls[doc].split("/")[-2]
        data.append(Review(text=text, summary=summary_dict[movie_id],
                           init_score=score, movie_id=movie_id, sentiment=VAL_DICT[category]))
    print("finished with: %s - %s" % (path_dir, category))
    return data


def get_data_from_csv(path):
    df = pd.read_csv(path)
    data = []
    for i, d in df.iterrows():
        data.append(Review(init_score=d[0], movie_id=d[1], sentiment=d[2], summary=str(d[3]), text=d[4]))
    return data


def get_posts(path, csv_path=None):
    summary_df = pd.read_csv(DATA_PATH+path+SUMMARIES_PATH)
    summary_dict = {str(m): str(s) for m, s in zip(summary_df['movie'].values, summary_df['summary'].values)}
    pos_data = load_data_per_category(path, 'pos', summary_dict)
    neg_data = load_data_per_category(path, 'neg', summary_dict)
    data = pos_data + neg_data
    if csv_path:
        translated = get_data_from_csv(csv_path)
        data = data + translated
    random.shuffle(data)
    return data


def split_data(data, split_point):
    """
    splits the data (for train and test)
    :param data: a list of tweets
    :param split_point: the point of splitting
    :return: two lists of tweets
    """
    return data[0:split_point], data[split_point:]


def generate_data(emb_size, min_freq=1, min_author_freq=3,
                  max_len=1000):
    """
    generates all necessary components for training and evaluation (posts, embedding matrix, dictionaries and batches
    :param emb_size: to size of word embeddings
    :param min_freq: how many times a word must be found in data in order t not being considered as unknown
    :param min_author_freq: least number of posts of the author in the dataset in order not to be consider unk
    even if its embedding is available
    :param max_len: the padding size of text
    :return: train_posts, test_posts, w2i, emb_matrix, train_batches, test_batches
    """

    train_posts = get_posts('train', csv_path=TRANSLATIONS_PATH)
    test_posts = get_posts('test')
    posts = train_posts + test_posts
    print('reviews for training:', len(train_posts))
    print('reviews for testing:', len(test_posts))
    embeddings_dict = get_embeddings(size=emb_size)
    w2i = create_final_dictionary(reviews=posts, min_freq=min_freq, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN,
                                  embeddings_dict=embeddings_dict)

    emb_matrix = get_embeddings_matrix(w2i, embeddings_dict, size=emb_size)
    m2i = create_movie_dictionary(train_posts, min_author_freq)
    train_batches = create_batches(train_posts, w2i, m2i, max_len=max_len)
    test_batches = create_batches(test_posts, w2i, m2i, max_len=max_len)

    return {'train_posts': train_posts, 'test posts': test_posts, 'w2i':w2i,
            'm2i': m2i, 'emb_matrix': emb_matrix,
            'train_batches': train_batches, 'test_batches': test_batches}


