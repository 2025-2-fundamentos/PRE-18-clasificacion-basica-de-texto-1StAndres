"""Entrena y guarda un vectorizador y un clasificador usados en los tests.

Este script carga `files/input/sentences.csv.zip`, entrena un
`TfidfVectorizer` y un `LogisticRegression`, y guarda los objetos
serializados en la carpeta `homework/` como `vectorizer.pkl` y `clf.pkl`.
"""
from __future__ import annotations

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(repo_root, os.pardir, "files", "input", "sentences.csv.zip")

    # Leer el csv comprimido
    df = pd.read_csv(csv_path, compression="zip", index_col=False)

    # Obtener textos y etiquetas
    X_texts = df["phrase"].astype(str).values
    y = df["target"].values

    # Vectorizador y clasificador sencillos pero eficaces
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(X_texts)

    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X, y)

    # Guardar objetos
    vectorizer_path = os.path.join(repo_root, "vectorizer.pkl")
    clf_path = os.path.join(repo_root, "clf.pkl")

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(clf_path, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved classifier to: {clf_path}")


if __name__ == "__main__":
    main()
