"""As métricas do dicionário reverso, e o protocolo que as torna honestas.

**recall@5 é a métrica do produto.** A tela mostra cinco candidatos, então a
pergunta que importa é "a resposta está na tela?", não "acertou de primeira?".
recall@1 e MRR entram para descrever *como* ela está lá — no topo ou no fim da
lista —, mas quem decide se o app serve é o recall@5.

**leave-one-articulator-out.** Com três pessoas gravando tudo, qualquer divisão
aleatória entre treino e teste põe o mesmo articulador dos dois lados, e a
métrica passa a medir memorização de pessoa. É o mesmo erro que a fase 1 cometeu
com o L↔G e pagou caro: o modelo aprendeu de que *fonte* vinha a amostra em vez
da forma da mão. Aqui o rodízio é por pessoa: indexa com duas, consulta com a
terceira, três vezes.

Números daqui não se comparam com acurácia de classificador fechado. A
referência da área para busca em vocabulário grande é 50,8% de MRR em 10.235
sinais (arXiv:2502.20171) — um recall@5 alto aqui é resultado bom, não consolo.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import catalogo
from .dicionario import Candidato, Dicionario


@dataclass(frozen=True)
class Metricas:
    """O placar de um rodízio."""

    consultas: int
    recall_1: float
    recall_5: float
    mrr: float

    def __str__(self) -> str:
        return (
            f"{self.consultas:5d} consultas   "
            f"recall@1 {self.recall_1:6.1%}   "
            f"recall@5 {self.recall_5:6.1%}   "
            f"MRR {self.mrr:6.1%}"
        )


def posicao(candidatos: list[Candidato], esperado: str) -> int | None:
    """Em que lugar da lista o sinal certo apareceu, contando de 1.

    None quando ele não apareceu. A comparação usa `catalogo.chave`, então
    `AMANHÃ` e `AMANHA` contam como acerto — a diferença é de quem digitou o
    nome do arquivo, não do modelo.
    """
    alvo = catalogo.chave(esperado)
    for i, candidato in enumerate(candidatos, start=1):
        if catalogo.chave(candidato.rotulo) == alvo:
            return i
    return None


def agregar(posicoes: list[int | None]) -> Metricas:
    """Posições de acerto → placar.

    Consulta que não achou nada conta como erro, nunca como ausente: descartar
    o que falhou é a forma mais fácil de inventar um número bonito.
    """
    total = len(posicoes)
    if total == 0:
        return Metricas(consultas=0, recall_1=0.0, recall_5=0.0, mrr=0.0)

    acertos_1 = sum(1 for p in posicoes if p == 1)
    acertos_5 = sum(1 for p in posicoes if p is not None and p <= 5)
    reciprocos = sum(1.0 / p for p in posicoes if p is not None)

    return Metricas(
        consultas=total,
        recall_1=acertos_1 / total,
        recall_5=acertos_5 / total,
        mrr=reciprocos / total,
    )


def leave_one_articulator_out(
    dicionario: Dicionario,
    k: int = 5,
) -> dict[str, Metricas]:
    """Um rodízio por articulador, mais a linha agregada em `"total"`.

    Args:
        dicionario: todos os protótipos, com a fonte de cada um.
        k: quantos candidatos a busca devolve. Precisa ser >= 5 para que o
            recall@5 signifique o que diz.

    Raises:
        ValueError: com menos de dois articuladores não há rodízio possível, e
            devolver um número aqui seria pior que recusar.
    """
    if k < 5:
        raise ValueError("k deve ser >= 5 para que recall@5 seja medível")

    fontes = dicionario.fontes
    if len(fontes) < 2:
        raise ValueError(
            f"leave-one-articulator-out precisa de >= 2 articuladores, "
            f"encontrado(s) {len(fontes)}: {fontes}"
        )

    por_fonte: dict[str, Metricas] = {}
    todas: list[int | None] = []

    for fonte in fontes:
        base, consultas = dicionario.separar_fonte(fonte)
        posicoes = [
            posicao(base.buscar(representacao, k=k), rotulo)
            for representacao, rotulo in consultas
        ]
        por_fonte[fonte] = agregar(posicoes)
        todas.extend(posicoes)

    por_fonte["total"] = agregar(todas)
    return por_fonte
