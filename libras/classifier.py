"""Carrega o modelo treinado e prediz a letra de um vetor de landmarks."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from . import config


class ModeloAusente(FileNotFoundError):
    """Erro com instrução de como resolver, em vez de stacktrace."""


class Classificador:
    """Envolve o pipeline do scikit-learn salvo por `training/train.py`."""

    def __init__(self, caminho: Path | None = None):
        caminho = caminho or config.MODELO_CLASSIFICADOR

        if not caminho.exists():
            raise ModeloAusente(
                f"Classificador nao encontrado em {caminho}.\n"
                "Treine o modelo primeiro:\n"
                "  python training/prepare_dataset.py\n"
                "  python training/collect.py\n"
                "  python training/train.py"
            )

        pacote = joblib.load(caminho)
        self._modelo = pacote["modelo"]
        self.letras: list[str] = list(pacote["letras"])
        self.acuracia: float = float(pacote.get("acuracia", 0.0))
        self.nome: str = pacote.get("nome", "desconhecido")

    def prever(self, vetor: np.ndarray) -> tuple[str, float]:
        """Prediz a letra de um único vetor (63,).

        Returns:
            (letra, confiança entre 0 e 1).
        """
        probabilidades = self._modelo.predict_proba(vetor.reshape(1, -1))[0]
        indice = int(np.argmax(probabilidades))
        return self.letras[indice], float(probabilidades[indice])
