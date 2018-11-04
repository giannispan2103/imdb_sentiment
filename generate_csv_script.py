import urllib2
import pandas as pd


def get_summary(url):
    # url = "http://www.imdb.com/title/tt0893406/" #.strip()
    try:
        response = urllib2.urlopen(url)
        page_source = response.read()
        summary = page_source.split("<div class=\"inline canwrap\">")[1].split("</span>")[0].split("<span>")[1].strip()
        response.close()
    except:
        print("exception:"+url)
        summary = ""
    return summary


def get_all_summaries(raw_urls):
    summaries = []
    movie_ids = []
    for i, r_url in enumerate(raw_urls, start=1):
        url = r_url.replace("/usercomments", "")
        movie_id = url.split("/")[-1]
        summary = get_summary(url)
        movie_ids.append(movie_id.strip())
        summaries.append(summary)
        if i % 100 == 0:
            print("finished with %d summaries" % i)
    return summaries, movie_ids


def generate_csv(path):
    full_path = "../input/aclImdb/%s/" % path
    print(path)
    with open(full_path + "urls_pos.txt", 'r') as f:
        urls_pos = f.readlines()
        print(len(urls_pos))
    with open(full_path + "urls_neg.txt", 'r') as f:
        urls_neg = f.readlines()
        print(len(urls_neg))

    urls = urls_neg + urls_pos
    urls = list(set(urls))

    summaries, movies = get_all_summaries(urls)

    df = pd.DataFrame({"movie": movies, "summary": summaries})
    df.to_csv(full_path+"summaries.csv", index=False)


if __name__ == "__main__":
    generate_csv("train")
    generate_csv("test")

