"""TF-IDF plus logistic-regression text classification baseline."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main() -> None:
    rows = [
        ("payment failed and I was charged twice", "billing"),
        ("please refund the duplicate card charge", "billing"),
        ("my invoice total is incorrect", "billing"),
        ("why did my subscription renew", "billing"),
        ("the app crashes when I open settings", "technical"),
        ("login page keeps showing an error", "technical"),
        ("the upload button is not working", "technical"),
        ("I cannot reset my password", "technical"),
        ("how do I change my delivery address", "account"),
        ("I need to update my email address", "account"),
        ("please close my account", "account"),
        ("where can I change notification settings", "account"),
    ] * 8
    df = pd.DataFrame(rows, columns=["text", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, stratify=df["label"], random_state=42
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    examples = [
        "I have been billed two times this month",
        "the software freezes after I log in",
        "I want to update the email on my profile",
    ]
    for text, label in zip(examples, model.predict(examples)):
        print(f"{label:10s} <- {text}")


if __name__ == "__main__":
    main()
