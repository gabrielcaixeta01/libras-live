"""Carrega o modelo treinado e prediz a letra de um vetor de landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .. import config


class ModeloAusente(FileNotFoundError):
    """Erro com instrução de como resolver, em vez de stacktrace."""


@dataclass(frozen=True)
class Predicao:
    """O palpite do modelo para um frame.

    `aceita` é a parte que importa. O classificador é fechado nas letras que
    viu treinando: dado qualquer mão, ele devolve a mais parecida entre elas,
    nunca "não sei". Se o modelo conhece 15 letras e você faz um F, sai a mais
    próxima das 15 — e às vezes com confiança alta. O limiar não resolve isso
    por completo, mas transforma o caso comum num "?" honesto na tela em vez de
    uma letra errada silenciosa no seu texto.
    """

    letra: str
    confianca: float
    aceita: bool

    @property
    def rotulo(self) -> str:
        """O que a UI mostra: a letra, ou "?" quando ficou abaixo do limiar."""
        return self.letra if self.aceita else "?"


class Classificador:
    """Envolve o pipeline do scikit-learn salvo por `training/train.py`."""

    def __init__(
        self,
        caminho: Path | None = None,
        limiar_rejeicao: float = config.CONFIANCA_REJEICAO,
    ):
        caminho = caminho or config.MODELO_CLASSIFICADOR

        if not caminho.exists():
            raise ModeloAusente(
                f"Classificador nao encontrado em {caminho}.\n"
                "Treine o modelo primeiro:\n"
                "  python training/prepare_dataset.py\n"
                "  python training/collect.py\n"
                "  python training/train.py"
            )

        self._instalar(joblib.load(caminho), limiar_rejeicao)

    @classmethod
    def de_pacote(
        cls,
        pacote: dict,
        limiar_rejeicao: float = config.CONFIANCA_REJEICAO,
    ) -> "Classificador":
        """Constrói a partir do dicionário já carregado — usado nos testes."""
        instancia = object.__new__(cls)
        instancia._instalar(pacote, limiar_rejeicao)
        return instancia

    def _instalar(self, pacote: dict, limiar_rejeicao: float) -> None:
        self._modelo = pacote["modelo"]
        self.letras: list[str] = list(pacote["letras"])
        self.acuracia: float = float(pacote.get("acuracia", 0.0))
        self.macro_f1: float = float(pacote.get("macro_f1", 0.0))
        self.nome: str = pacote.get("nome", "desconhecido")
        self.limiar_rejeicao = limiar_rejeicao

    @property
    def letras_ausentes(self) -> list[str]:
        """Letras do alfabeto que este modelo nunca viu — e nunca vai prever."""
        return sorted(set(config.ALFABETO) - set(self.letras))

    def prever(self, vetor: np.ndarray) -> Predicao:
        """Prediz a letra de um único vetor (63,)."""
        probabilidades = self._modelo.predict_proba(vetor.reshape(1, -1))[0]
        indice = int(np.argmax(probabilidades))
        confianca = float(probabilidades[indice])

        return Predicao(
            letra=self.letras[indice],
            confianca=confianca,
            aceita=confianca >= self.limiar_rejeicao,
        )
