"""Quando o dicionário deve dizer "não reconheci" em vez de mostrar cinco palavras.

O índice tem 163 sinais e a Libras tem milhares: quase tudo que você pode
sinalizar está fora dele. A busca não sabe disso — ela sempre devolve os cinco
mais próximos, com a mesma cara de resposta que teria se estivesse certa. É a
versão em sinais do problema que `CONFIANCA_REJEICAO` resolve no alfabeto, e sem
alguma regra aqui o app mente com convicção.

**Duas regras possíveis, e a diferença entre elas é o que este módulo existe
para medir.**

- *Distância*: recusa quando o melhor candidato está longe. Simples, e o sinal é
  fraco — a distância do cosseno não é confiança, e uma consulta difícil que
  acertou pode estar tão longe quanto uma consulta impossível.
- *Margem*: recusa quando o primeiro e o segundo candidatos estão empatados.
  Mede outra coisa — não "isto parece com alguma coisa?" e sim "isto parece com
  **uma** coisa?". Um sinal que o dicionário não tem cai no meio de um monte de
  protótipos parecidos e não destaca nenhum; um que ele tem destaca o certo.

Qual das duas paga é pergunta empírica, e a resposta fica em
`models/relatorio_encoder.txt`, calibrada nos mesmos rodízios que medem o
recall. Aqui ficam a regra e a calibração — lógica pura, testável sem torch,
sem câmera e sem dicionário.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Quanto dos acertos o limiar tem que preservar. O custo dos dois erros é
# assimétrico e o número sai daí: mostrar cinco palavras erradas custa quase
# nada num dicionário — você olha, não reconhece nenhuma e refaz o sinal —,
# enquanto recusar uma consulta que teria acertado apaga a resposta que a pessoa
# procurava. Maximizar o J de Youden ignora essa assimetria e foi medido em
# 0,2878, um corte que guardava 37% dos acertos: mais honesto e muito menos útil.
COBERTURA_MINIMA = 0.95


@dataclass(frozen=True)
class Consulta:
    """O que uma consulta deixou para trás, e se ela tinha resposta.

    Attributes:
        primeira: distância do melhor candidato.
        segunda: distância do segundo. `inf` quando só havia um candidato — é o
            caso de margem infinita, ou seja, o mais certo possível.
        acertou: a resposta certa estava entre os `k` mostrados.
    """

    primeira: float
    segunda: float
    acertou: bool

    @property
    def margem(self) -> float:
        """Quanto o primeiro candidato ganhou do segundo. Grande é bom."""
        if not math.isfinite(self.segunda):
            return math.inf
        return self.segunda - self.primeira


@dataclass(frozen=True)
class Corte:
    """Onde cortar, e o que se ganha e se perde cortando ali."""

    criterio: str            # "distancia" ou "margem"
    limiar: float
    aceitos_certos: int      # aceitou, e a resposta estava na lista
    aceitos_errados: int     # aceitou, e não estava — é a mentira que o corte tira
    recusados_certos: int    # recusou uma consulta que teria acertado
    recusados_errados: int   # recusou, e fez bem

    @property
    def precisao(self) -> float:
        """Das listas que o app mostra, quantas contêm a resposta."""
        mostradas = self.aceitos_certos + self.aceitos_errados
        return self.aceitos_certos / mostradas if mostradas else 0.0

    @property
    def cobertura(self) -> float:
        """Dos acertos possíveis, quantos sobrevivem ao corte."""
        possiveis = self.aceitos_certos + self.recusados_certos
        return self.aceitos_certos / possiveis if possiveis else 0.0

    @property
    def recusa_correta(self) -> float:
        """Do que não tinha resposta, quanto o app teve a honestidade de recusar."""
        sem_resposta = self.aceitos_errados + self.recusados_errados
        return self.recusados_errados / sem_resposta if sem_resposta else 0.0


def _valor(consulta: Consulta, criterio: str) -> float:
    if criterio == "distancia":
        return consulta.primeira
    if criterio == "margem":
        return consulta.margem
    raise ValueError(f"critério desconhecido: {criterio!r}")


def calibrar(
    consultas: list[Consulta],
    criterio: str = "distancia",
    cobertura_minima: float = COBERTURA_MINIMA,
) -> Corte | None:
    """O corte que preserva `cobertura_minima` dos acertos, e o que ele compra.

    Em *distância* aceita-se o que está **abaixo** do corte; em *margem*, o que
    está **acima** dele. Nos dois casos o corte é o quantil das consultas que
    acertaram, escolhido para deixar passar a fração pedida delas.

    As consultas sem resposta possível — os sinais fora do vocabulário indexado —
    não escolhem o corte, mas entram na contagem: são elas que medem o que ele
    compra. Sem elas o número seria escolhido olhando só para quem tinha resposta.

    Returns:
        None quando nenhuma consulta acertou: não há de onde tirar o quantil, e
        um corte inventado seria pior que nenhum.
    """
    if not 0.0 < cobertura_minima <= 1.0:
        raise ValueError(f"cobertura_minima fora de (0, 1]: {cobertura_minima}")

    certas = sorted(_valor(c, criterio) for c in consultas if c.acertou)
    if not certas:
        return None

    maior_e_melhor = criterio == "margem"
    if maior_e_melhor:
        # A margem é boa quando é grande, então o quantil vem da outra ponta.
        posicao = max(0, len(certas) - math.ceil(cobertura_minima * len(certas)))
        limiar = certas[min(posicao, len(certas) - 1)]
    else:
        posicao = min(len(certas) - 1, math.ceil(cobertura_minima * len(certas)) - 1)
        limiar = certas[max(posicao, 0)]

    def aceita(consulta: Consulta) -> bool:
        valor = _valor(consulta, criterio)
        return valor >= limiar if maior_e_melhor else valor <= limiar

    aceitos_certos = sum(1 for c in consultas if c.acertou and aceita(c))
    aceitos_errados = sum(1 for c in consultas if not c.acertou and aceita(c))

    return Corte(
        criterio=criterio,
        limiar=float(limiar),
        aceitos_certos=aceitos_certos,
        aceitos_errados=aceitos_errados,
        recusados_certos=len(certas) - aceitos_certos,
        recusados_errados=(len(consultas) - len(certas)) - aceitos_errados,
    )


def recusar(
    candidatos: list,
    limiar_distancia: float | None,
    limiar_margem: float | None,
) -> bool:
    """A regra como o app a aplica: esta lista deve virar "não reconheci"?

    Os dois critérios valem juntos, e qualquer um deles basta para recusar. Eles
    pegam falhas diferentes — longe de tudo, ou empatado entre coisas — e um
    `None` desliga o seu.

    Lista vazia nunca é recusa: não há o que recusar, e quem chama já sabe que
    não tem nada a mostrar.
    """
    if not candidatos:
        return False

    if limiar_distancia is not None and candidatos[0].distancia > limiar_distancia:
        return True

    if limiar_margem is not None and len(candidatos) >= 2:
        if candidatos[1].distancia - candidatos[0].distancia < limiar_margem:
            return True

    return False
