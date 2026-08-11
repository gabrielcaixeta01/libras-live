"""Treina o classificador de letras a partir dos landmarks.

Junta a base pública (`prepare_dataset.py`) com as suas gravações
(`collect.py`), compara alguns modelos por validação cruzada e salva o melhor.

    python training/train.py

Duas decisões que mudam o que o número final significa:

**Macro-F1, não acurácia.** As classes são muito desiguais — B tem quinze vezes
mais amostras que N. Acurácia global é dominada pelas classes fartas: um modelo
que erre N inteiro perde menos de 1% de acurácia. Macro-F1 dá o mesmo peso a
cada letra, então errar N custa o mesmo que errar B. É a métrica que responde à
pergunta que interessa: "ele reconhece o alfabeto?", não "ele acerta muito?".

**Aumento dentro de cada fold, nunca antes de separar.** As amostras sintéticas
do `augment` são derivadas das reais. Se o aumento vier antes da separação, uma
variação da mesma mão cai no treino e outra no teste, e a métrica sobe sem que o
modelo tenha melhorado. Aqui o teste é sempre só de amostras reais.
"""

from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from libras import config
from libras.augment import equilibrar

DIR_COLETADOS = config.DIR_DADOS / "coletados"
DATASET_PUBLICO = config.DIR_DADOS / "dataset_publico.npz"

FOLDS = 5
SEMENTE = 42

CANDIDATOS = {
    "random_forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEMENTE,
        n_jobs=-1,
    ),
    # MLP não aceita class_weight; para ele o equilíbrio vem só do aumento.
    "mlp": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=1200,
            early_stopping=True,
            random_state=SEMENTE,
        ),
    ),
    # O app precisa de probabilidade calibrada, não só da classe — o
    # estabilizador filtra por confiança. SVC(probability=True) foi depreciado
    # no sklearn 1.9; CalibratedClassifierCV é o substituto.
    "svm_rbf": make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(C=10, gamma="scale", class_weight="balanced", random_state=SEMENTE),
            ensemble=False,
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


def _avaliar_fold(modelo, X, y, treino, validacao, semente) -> float:
    """Um fold: aumenta o treino, ajusta um clone e mede no que ficou de fora."""
    X_aug, y_aug = equilibrar(X[treino], y[treino], semente=semente)

    candidato = clone(modelo)
    candidato.fit(X_aug, y_aug)

    predito = candidato.predict(X[validacao])
    return float(f1_score(y[validacao], predito, average="macro", zero_division=0))


def validar(modelo, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Macro-F1 por validação cruzada, aumentando só a parte de treino de cada fold.

    Os folds rodam em paralelo: com o aumento, cada um treina em ~9 mil amostras,
    e o SVM sozinho levaria meia hora em série.

    Returns:
        (média, desvio padrão) do macro-F1 entre os folds.
    """
    dobras = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEMENTE)

    scores = joblib.Parallel(n_jobs=FOLDS)(
        joblib.delayed(_avaliar_fold)(modelo, X, y, treino, validacao, SEMENTE + indice)
        for indice, (treino, validacao) in enumerate(dobras.split(X, y))
    )

    return float(np.mean(scores)), float(np.std(scores))


def _tabela_confusao(y_real, y_predito, classes) -> str:
    """Matriz de confusão em texto, com as classes nas linhas e colunas."""
    matriz = confusion_matrix(y_real, y_predito, labels=classes)
    largura = max(4, max(len(str(c)) for c in classes) + 1)

    linhas = [" " * largura + "".join(f"{c:>{largura}}" for c in classes)]
    for nome, linha in zip(classes, matriz):
        linhas.append(
            f"{nome:>{largura}}" + "".join(f"{n:>{largura}}" for n in linha)
        )
    return "\n".join(linhas)


def _piores_classes(y_real, y_predito, classes, limite: int = 5) -> list[tuple[str, float]]:
    """As letras com menor F1 — onde vale gravar amostras suas."""
    f1_por_classe = f1_score(
        y_real, y_predito, labels=classes, average=None, zero_division=0
    )
    pares = sorted(zip(classes, f1_por_classe), key=lambda par: par[1])
    return [(str(c), float(f)) for c, f in pares[:limite]]


def _confusoes(y_real, y_predito, classes, limite: int = 5):
    """Os pares que o modelo mais troca."""
    matriz = confusion_matrix(y_real, y_predito, labels=classes).copy()
    np.fill_diagonal(matriz, 0)

    pares = [
        (int(matriz[i, j]), str(classes[i]), str(classes[j]))
        for i in range(len(classes))
        for j in range(len(classes))
        if matriz[i, j] > 0
    ]
    return sorted(pares, reverse=True)[:limite]


def main() -> int:
    X, y = carregar()

    contagem = Counter(y.tolist())
    print(f"\nTotal: {len(X)} amostras reais, {len(contagem)} classes")
    print("Por classe: " + "  ".join(f"{l}:{n}" for l, n in sorted(contagem.items())))

    faltando = sorted(set(config.ALFABETO) - set(contagem))
    if faltando:
        print(f"\nAVISO: sem dados para {' '.join(faltando)}.")
        print("O modelo nunca vai prever essas letras.")

    escassas = [l for l, n in contagem.items() if n < 20]
    if escassas:
        print(f"AVISO: poucas amostras em {' '.join(sorted(escassas))}.")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE
    )

    print(
        f"\nAumento: cada classe do treino vai a {config.AUG_ALVO_POR_CLASSE} amostras."
    )
    print("O conjunto de teste fica só com amostras reais.")

    print(f"\nComparando modelos (macro-F1, {FOLDS} folds):")
    melhor_nome, melhor_score = None, -1.0

    for nome, modelo in CANDIDATOS.items():
        media, desvio = validar(modelo, X_treino, y_treino)
        print(f"  {nome:>14}: {media:.1%} (+/- {desvio:.1%})")
        if media > melhor_score:
            melhor_nome, melhor_score = nome, media

    print(f"\nVencedor: {melhor_nome}")

    X_aug, y_aug = equilibrar(X_treino, y_treino, semente=SEMENTE)
    print(f"Treino final: {len(X_treino)} reais -> {len(X_aug)} com aumento")

    modelo = clone(CANDIDATOS[melhor_nome])
    modelo.fit(X_aug, y_aug)

    y_predito = modelo.predict(X_teste)
    classes = list(modelo.classes_)
    acuracia = float((y_predito == y_teste).mean())
    macro_f1 = float(f1_score(y_teste, y_predito, average="macro", zero_division=0))

    print("\nNo conjunto de teste (real, sem aumento):")
    print(f"  macro-F1: {macro_f1:.1%}   <- a métrica que importa")
    print(f"  acurácia: {acuracia:.1%}")

    piores = _piores_classes(y_teste, y_predito, classes)
    print("\nLetras com pior F1 (grave amostras suas destas):")
    for letra, valor in piores:
        print(f"  {letra}: {valor:.1%}  ({contagem.get(letra, 0)} amostras reais)")

    confusoes = _confusoes(y_teste, y_predito, classes)
    if confusoes:
        print("\nLetras mais confundidas (real -> predito):")
        for n, real, predito in confusoes:
            print(f"  {real} -> {predito}: {n}x")

    _salvar_relatorio(
        melhor_nome, contagem, X_treino, X_aug, y_teste, y_predito, classes,
        acuracia, macro_f1, piores, confusoes,
    )

    config.MODELO_CLASSIFICADOR.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": modelo,
            "letras": classes,
            "acuracia": acuracia,
            "macro_f1": macro_f1,
            "nome": melhor_nome,
            "amostras_reais": dict(contagem),
        },
        config.MODELO_CLASSIFICADOR,
    )
    print(f"\nModelo salvo em {config.MODELO_CLASSIFICADOR}")
    print(f"Relatório salvo em {config.RELATORIO_TREINO}")
    print("Rode o app com: python -m libras.app")
    return 0


def _salvar_relatorio(
    nome, contagem, X_treino, X_aug, y_teste, y_predito, classes,
    acuracia, macro_f1, piores, confusoes,
) -> None:
    """Grava tudo em disco: o terminal rola, o arquivo fica para comparar depois."""
    saida = io.StringIO()

    saida.write("Relatório de treino — libras-live\n")
    saida.write("=" * 60 + "\n\n")
    saida.write(f"Modelo escolhido: {nome}\n")
    saida.write(f"Macro-F1 (teste real): {macro_f1:.4f}\n")
    saida.write(f"Acurácia (teste real): {acuracia:.4f}\n\n")

    saida.write(f"Amostras reais por classe ({sum(contagem.values())} no total):\n")
    for letra, quantas in sorted(contagem.items()):
        saida.write(f"  {letra}: {quantas}\n")

    faltando = sorted(set(config.ALFABETO) - set(contagem))
    if faltando:
        saida.write(f"\nLetras sem nenhum dado: {' '.join(faltando)}\n")

    saida.write(f"\nTreino: {len(X_treino)} reais -> {len(X_aug)} com aumento\n")
    saida.write(
        f"Aumento: rotação ±{config.AUG_ROTACAO_GRAUS}°, ruído {config.AUG_RUIDO}, "
        f"profundidade ±{config.AUG_PROFUNDIDADE:.0%}, "
        f"alvo {config.AUG_ALVO_POR_CLASSE}/classe\n"
    )

    saida.write("\n" + "-" * 60 + "\n")
    saida.write("Relatório por classe\n\n")
    saida.write(classification_report(y_teste, y_predito, zero_division=0))

    saida.write("\n" + "-" * 60 + "\n")
    saida.write("Matriz de confusão (linha = real, coluna = predito)\n\n")
    saida.write(_tabela_confusao(y_teste, y_predito, classes))

    saida.write("\n\n" + "-" * 60 + "\n")
    saida.write("Letras com pior F1\n\n")
    for letra, valor in piores:
        saida.write(f"  {letra}: {valor:.4f}  ({contagem.get(letra, 0)} amostras reais)\n")

    if confusoes:
        saida.write("\nPares mais confundidos (real -> predito)\n\n")
        for n, real, predito in confusoes:
            saida.write(f"  {real} -> {predito}: {n}x\n")

    config.RELATORIO_TREINO.parent.mkdir(parents=True, exist_ok=True)
    config.RELATORIO_TREINO.write_text(saida.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
