from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def candidates():
    word = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    return {
        "majority_baseline": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "tfidf_word_logreg": Pipeline([("v", word()), ("clf", LogisticRegression(max_iter=2000, C=10))]),
        "tfidf_char_logreg": Pipeline([("v", char()), ("clf", LogisticRegression(max_iter=2000, C=10))]),
        "tfidf_word+char_svm": Pipeline([("v", FeatureUnion([("w", word()), ("c", char())])),
                                         ("clf", LinearSVC(C=1.0))]),
    }
