"""Treina o classificador de letras a partir dos landmarks.

Junta a base pública (`prepare_dataset.py`) com as suas gravações
(`collect.py`), compara alguns modelos por validação cruzada e salva o melhor.

    python training/train.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from libras import config

DIR_COLETADOS = config.DIR_DADOS / "coletados"
DATASET_PUBLICO = config.DIR_DADOS / "dataset_publico.npz"

CANDIDATOS = {
    "random_forest": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "mlp": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=1200,
            early_stopping=True,
            random_state=42,
        ),
    ),
    # O app precisa de probabilidade calibrada, não só da classe — o
    # estabilizador filtra por confiança. SVC(probability=True) foi depreciado
    # no sklearn 1.9; CalibratedClassifierCV é o substituto.
    "svm_rbf": make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(C=10, gamma="scale", random_state=42), ensemble=False
        ),
    ),
}


def carregar() -> tuple[np.ndarray, np.ndarray]:
    """Junta as duas fontes. Ambas já são vetores de 63 floats."""
    vetores: list[np.ndarray] = []
    rotulos: list[np.ndarray] = []

    if DATASET_PUBLICO.exists():
        dados = np.load(DATASET_PUBLICO, allow_pickle=True)
        vetores.append(dados["X"])
        rotulos.append(dados["y"])
        print(f"Base pública: {len(dados['X'])} amostras")
    else:
        print(f"Base pública ausente ({DATASET_PUBLICO}).")
        print("Rode: python training/prepare_dataset.py")

    if DIR_COLETADOS.exists():
        for caminho in sorted(DIR_COLETADOS.glob("*.npy")):
            amostras = np.load(caminho)
            vetores.append(amostras)
            rotulos.append(np.full(len(amostras), caminho.stem))
        gravadas = sorted(p.stem for p in DIR_COLETADOS.glob("*.npy"))
        if gravadas:
            print(f"Gravações próprias: {' '.join(gravadas)}")

    if not vetores:
        raise SystemExit(
            "Nenhum dado encontrado. Rode prepare_dataset.py e/ou collect.py."
        )

    return np.vstack(vetores).astype(np.float32), np.concatenate(rotulos)


def main() -> int:
    X, y = carregar()

    contagem = Counter(y)
    print(f"\nTotal: {len(X)} amostras, {len(contagem)} classes")
    print("Por classe: " + "  ".join(f"{l}:{n}" for l, n in sorted(contagem.items())))

    faltando = sorted(set(config.ALFABETO) - set(contagem))
    if faltando:
        print(f"\nAVISO: sem dados para {' '.join(faltando)}.")
        print("O modelo nunca vai prever essas letras.")

    escassas = [l for l, n in contagem.items() if n < 20]
    if escassas:
        print(f"AVISO: poucas amostras em {' '.join(sorted(escassas))}.")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nComparando modelos (validação cruzada, 5 folds):")
    melhor_nome, melhor_score = None, -1.0

    for nome, modelo in CANDIDATOS.items():
        scores = cross_val_score(modelo, X_treino, y_treino, cv=5, n_jobs=-1)
        media = float(scores.mean())
        print(f"  {nome:>14}: {media:.1%} (+/- {scores.std():.1%})")
        if media > melhor_score:
            melhor_nome, melhor_score = nome, media

    print(f"\nVencedor: {melhor_nome}")
    modelo = CANDIDATOS[melhor_nome]
    modelo.fit(X_treino, y_treino)

    y_predito = modelo.predict(X_teste)
    acuracia = float((y_predito == y_teste).mean())
    print(f"Acurácia no conjunto de teste: {acuracia:.1%}\n")
    print(classification_report(y_teste, y_predito, zero_division=0))

    _reportar_confusoes(y_teste, y_predito, modelo.classes_)

    config.MODELO_CLASSIFICADOR.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": modelo,
            "letras": list(modelo.classes_),
            "acuracia": acuracia,
            "nome": melhor_nome,
        },
        config.MODELO_CLASSIFICADOR,
    )
    print(f"\nModelo salvo em {config.MODELO_CLASSIFICADOR}")
    print("Rode o app com: python -m libras.app")
    return 0


def _reportar_confusoes(y_teste, y_predito, classes, limite: int = 5) -> None:
    """Os pares que o modelo mais troca — é onde vale gravar mais amostras."""
    matriz = confusion_matrix(y_teste, y_predito, labels=classes)
    np.fill_diagonal(matriz, 0)

    pares = [
        (matriz[i, j], classes[i], classes[j])
        for i in range(len(classes))
        for j in range(len(classes))
        if matriz[i, j] > 0
    ]
    if not pares:
        print("Nenhuma confusão no conjunto de teste.")
        return

    print("Letras mais confundidas (real -> predito):")
    for n, real, predito in sorted(pares, reverse=True)[:limite]:
        print(f"  {real} -> {predito}: {n}x")


if __name__ == "__main__":
    raise SystemExit(main())
