"""O dicionário reverso: uma consulta entra, cinco palavras saem.

Um dicionário de Libras é indexado por português. Se você vê um sinal que não
conhece, não tem como procurar — você não sabe o nome dele, que é justamente o
que está tentando descobrir. Este módulo inverte o índice: a chave passa a ser a
**forma**.

Isso também é o que torna o top-5 honesto em vez de uma desculpa. Num tradutor,
cinco respostas é uma resposta errada. Num dicionário, cinco candidatos é uma
resposta boa — você reconhece a certa quando a vê.

**Protótipos separados, nunca a média.** Cada sinal do V-LIBRASIL tem três
gravações, uma por articulador, e elas ficam as três. A média inventaria um
quarto articulador que não existe, e apagaria justamente a variação entre
pessoas, que é a informação mais escassa numa base de três.

**Re-ancoragem.** `ancorar` acrescenta um protótipo seu e muda o resultado da
busca **sem retreinar nada**. É a mitigação estrutural do viés de três
articuladores — a mesma lição do L↔G da fase 1, resolvida por arquitetura em vez
de por mais coleta. Serve também para sinais que não estão nas 1.364 palavras.

A métrica entra por parâmetro para que o encoder neural, se ele vier, substitua
o DTW sem que o app perceba: sequência (T, 147) com DTW hoje, embedding (256,)
com cosseno depois.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import dtw

FONTE_USUARIO = "voce"


@dataclass(frozen=True)
class Candidato:
    """Uma linha do resultado da busca."""

    rotulo: str
    distancia: float
    similaridade: float
    fonte: str


def _metrica_dtw(consulta: np.ndarray, base: np.ndarray) -> np.ndarray:
    return dtw.distancias(consulta, base)


def _metrica_cosseno(consulta: np.ndarray, base: np.ndarray) -> np.ndarray:
    """1 - cosseno, para quando a representação for um embedding."""
    consulta = np.asarray(consulta, dtype=np.float32).reshape(-1)
    base = np.asarray(base, dtype=np.float32)

    normas = np.linalg.norm(base, axis=1) * np.linalg.norm(consulta)
    normas = np.maximum(normas, 1e-8)
    return 1.0 - (base @ consulta) / normas


METRICAS = {"dtw": _metrica_dtw, "cosseno": _metrica_cosseno}
DIMENSOES_ESPERADAS = {"dtw": 3, "cosseno": 2}


class Dicionario:
    """Protótipos rotulados e a busca por cima deles."""

    def __init__(
        self,
        representacoes: np.ndarray,
        rotulos: list[str],
        fontes: list[str],
        metrica: str = "dtw",
    ):
        if metrica not in METRICAS:
            raise ValueError(
                f"métrica desconhecida: {metrica!r}. Conhecidas: {sorted(METRICAS)}"
            )

        representacoes = np.asarray(representacoes, dtype=np.float32)
        esperado = DIMENSOES_ESPERADAS[metrica]
        if representacoes.ndim != esperado:
            raise ValueError(
                f"métrica {metrica!r} espera representações com {esperado} dimensões, "
                f"recebido shape {representacoes.shape}"
            )
        if not (len(representacoes) == len(rotulos) == len(fontes)):
            raise ValueError(
                f"desalinhado: {len(representacoes)} representações, "
                f"{len(rotulos)} rótulos, {len(fontes)} fontes"
            )

        self.metrica = metrica
        self._representacoes = representacoes
        self._rotulos = list(rotulos)
        self._fontes = list(fontes)

    # --- busca ---

    def buscar(self, consulta: np.ndarray, k: int = 5) -> list[Candidato]:
        """Os `k` sinais mais próximos, um por rótulo, do mais para o menos provável.

        Cada rótulo aparece uma vez só, com a distância do seu melhor protótipo:
        dois articuladores da mesma palavra não podem ocupar duas linhas da tela.
        """
        if len(self) == 0 or k < 1:
            return []

        distancias = np.asarray(
            METRICAS[self.metrica](consulta, self._representacoes), dtype=np.float64
        )

        melhor_por_rotulo: dict[str, int] = {}
        for i, rotulo in enumerate(self._rotulos):
            atual = melhor_por_rotulo.get(rotulo)
            if atual is None or distancias[i] < distancias[atual]:
                melhor_por_rotulo[rotulo] = i

        ordenados = sorted(melhor_por_rotulo.items(), key=lambda par: distancias[par[1]])

        return [
            Candidato(
                rotulo=rotulo,
                distancia=float(distancias[i]),
                similaridade=_similaridade(float(distancias[i])),
                fonte=self._fontes[i],
            )
            for rotulo, i in ordenados[:k]
        ]

    # --- re-ancoragem ---

    def ancorar(
        self,
        representacao: np.ndarray,
        rotulo: str,
        fonte: str = FONTE_USUARIO,
    ) -> None:
        """Acrescenta um protótipo. Vale para a busca seguinte, sem retreino."""
        representacao = np.asarray(representacao, dtype=np.float32)
        formato = self._representacoes.shape[1:]

        if len(self) and representacao.shape != formato:
            raise ValueError(
                f"esperado shape {formato}, recebido {representacao.shape}"
            )

        self._representacoes = (
            representacao[None]
            if len(self) == 0
            else np.concatenate([self._representacoes, representacao[None]])
        )
        self._rotulos.append(rotulo)
        self._fontes.append(fonte)

    def remover_fonte(self, fonte: str) -> int:
        """Descarta todos os protótipos de uma fonte. Devolve quantos saíram.

        É como se desfaz uma re-ancoragem ruim sem reconstruir o dicionário base.
        """
        manter = [i for i, f in enumerate(self._fontes) if f != fonte]
        removidos = len(self) - len(manter)

        self._representacoes = self._representacoes[manter]
        self._rotulos = [self._rotulos[i] for i in manter]
        self._fontes = [self._fontes[i] for i in manter]
        return removidos

    # --- estado ---

    def __len__(self) -> int:
        return len(self._representacoes)

    @property
    def vocabulario(self) -> list[str]:
        return sorted(set(self._rotulos))

    @property
    def fontes(self) -> list[str]:
        return sorted(set(self._fontes))

    @property
    def rotulos(self) -> list[str]:
        return list(self._rotulos)

    def separar_fonte(self, fonte: str) -> tuple["Dicionario", list[tuple]]:
        """Divide em (dicionário sem a fonte, consultas daquela fonte).

        É a operação que o leave-one-articulator-out precisa e a única forma
        honesta de avaliar uma base de três pessoas: indexa com duas, consulta
        com a terceira. Qualquer protocolo que deixe o mesmo articulador dos
        dois lados mede memorização.
        """
        dentro = [i for i, f in enumerate(self._fontes) if f != fonte]
        fora = [i for i, f in enumerate(self._fontes) if f == fonte]

        consultas = [(self._representacoes[i], self._rotulos[i]) for i in fora]
        return self._subconjunto(dentro), consultas

    def apenas(self, fonte: str) -> "Dicionario":
        """Só os protótipos de uma fonte — é o que se grava em disco ao sair."""
        return self._subconjunto([i for i, f in enumerate(self._fontes) if f == fonte])

    def juntar(self, outro: "Dicionario") -> "Dicionario":
        """Empilha dois dicionários. O da direita entra depois, na mesma métrica."""
        if outro.metrica != self.metrica:
            raise ValueError(
                f"métricas diferentes: {self.metrica!r} e {outro.metrica!r}"
            )
        if len(outro) == 0:
            return self._subconjunto(range(len(self)))
        if len(self) == 0:
            return outro._subconjunto(range(len(outro)))

        return Dicionario(
            representacoes=np.concatenate(
                [self._representacoes, outro._representacoes]
            ),
            rotulos=self._rotulos + outro._rotulos,
            fontes=self._fontes + outro._fontes,
            metrica=self.metrica,
        )

    def _subconjunto(self, indices) -> "Dicionario":
        indices = list(indices)
        return Dicionario(
            representacoes=self._representacoes[indices],
            rotulos=[self._rotulos[i] for i in indices],
            fontes=[self._fontes[i] for i in indices],
            metrica=self.metrica,
        )

    # --- persistência ---

    def salvar(self, caminho: str | Path) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            caminho,
            representacoes=self._representacoes,
            rotulos=np.array(self._rotulos),
            fontes=np.array(self._fontes),
            metrica=np.array(self.metrica),
        )

    @classmethod
    def carregar(cls, caminho: str | Path) -> "Dicionario":
        dados = np.load(Path(caminho), allow_pickle=False)
        return cls(
            representacoes=dados["representacoes"],
            rotulos=[str(r) for r in dados["rotulos"]],
            fontes=[str(f) for f in dados["fontes"]],
            metrica=str(dados["metrica"]),
        )


def _similaridade(distancia: float) -> float:
    """Distância → [0, 1] para a barra na tela.

    É monotônica e nada mais. **Não é probabilidade** e não deve ser lida como
    tal: 0,81 não quer dizer "81% de chance de ser CASA", quer dizer "foi o mais
    perto, e por esta margem". O que ordena a lista é a distância; isto só a
    desenha.
    """
    if not np.isfinite(distancia):
        return 0.0
    return float(1.0 / (1.0 + max(distancia, 0.0)))
