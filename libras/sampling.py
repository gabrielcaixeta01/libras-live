"""Aceita uma amostra nova só quando ela é diferente das que já foram gravadas.

A primeira coleta gravou 200 amostras por letra em cerca de oito segundos: a
30fps, as amostras saem antes de a mão ter tempo de mudar de ângulo. O resultado
mediu-se depois — K e Q ficaram com raio 0,12 no espaço dos landmarks, contra
0,64 de média nas letras da base pública. O modelo não aprendeu a letra,
aprendeu uma pose exata, e qualquer desvio dela na hora do uso vira outra letra.

A correção não é gravar por mais tempo, é gravar *coisas diferentes*: uma
amostra só entra se estiver a pelo menos `distancia_minima` de todas as que já
entraram. Isso força a mão a passear — girar o pulso, aproximar, inclinar — e
converte tempo de gravação em variedade de verdade em vez de repetição.

O limiar afrouxa sozinho quando nada é aceito por um tempo. Sem isso a coleta
poderia travar para sempre numa letra cuja pose não admite muita variação, e
uma coleta que não termina é pior que uma coleta apertada.
"""

from __future__ import annotations

import numpy as np

from .landmarks import TAMANHO_VETOR


class ColetorDiverso:
    """Junta `total` amostras mantendo uma distância mínima entre elas.

    É amostragem por disco de Poisson no espaço dos landmarks: cada amostra
    aceita "queima" uma vizinhança ao redor de si, então a nuvem final cobre a
    região em vez de se empilhar num ponto.
    """

    def __init__(
        self,
        total: int,
        distancia_minima: float,
        paciencia: float = 1.5,
        decaimento: float = 0.8,
        limiar_minimo: float = 0.02,
    ):
        if total < 1:
            raise ValueError("total deve ser >= 1")
        if distancia_minima <= 0:
            raise ValueError("distancia_minima deve ser > 0")

        self.total = total
        self.limiar = float(distancia_minima)
        self.paciencia = paciencia
        self.decaimento = decaimento
        self.limiar_minimo = limiar_minimo

        self._amostras: list[np.ndarray] = []
        self._ultima_aceita = 0.0

    # --- coleta ---

    def oferecer(self, vetor: np.ndarray, segundos: float) -> bool:
        """Propõe uma amostra. Devolve True se ela foi aceita.

        Args:
            vetor: landmarks normalizados (63,).
            segundos: instante do frame, para medir há quanto tempo nada entra.
        """
        vetor = np.asarray(vetor, dtype=np.float32)
        if vetor.shape != (TAMANHO_VETOR,):
            raise ValueError(f"esperado ({TAMANHO_VETOR},), recebido {vetor.shape}")

        if self.completo:
            return False

        if self._amostras:
            self._talvez_afrouxar(segundos)
            distancia = float(
                np.linalg.norm(np.array(self._amostras) - vetor, axis=1).min()
            )
            if distancia < self.limiar:
                return False

        self._amostras.append(vetor.copy())
        self._ultima_aceita = segundos
        return True

    def _talvez_afrouxar(self, segundos: float) -> None:
        """Baixa o limiar quando a coleta empaca — a variedade fácil acabou."""
        if segundos - self._ultima_aceita < self.paciencia:
            return

        self.limiar = max(self.limiar * self.decaimento, self.limiar_minimo)
        self._ultima_aceita = segundos

    # --- estado ---

    @property
    def amostras(self) -> np.ndarray:
        """As amostras aceitas, no formato que `train.py` espera."""
        if not self._amostras:
            return np.empty((0, TAMANHO_VETOR), dtype=np.float32)
        return np.array(self._amostras, dtype=np.float32)

    @property
    def completo(self) -> bool:
        return len(self._amostras) >= self.total

    @property
    def progresso(self) -> float:
        return len(self._amostras) / self.total

    @property
    def dispersao(self) -> float:
        """Raio médio da nuvem — a mesma medida que expôs o problema.

        É o número a observar durante a gravação: abaixo de ~0,3 a coleta está
        apertada demais e o modelo vai decorar a pose.
        """
        if len(self._amostras) < 2:
            return 0.0
        amostras = np.array(self._amostras)
        return float(np.linalg.norm(amostras - amostras.mean(0), axis=1).mean())

    def parado(self, segundos: float) -> bool:
        """True quando faz tempo que nada é aceito — a UI pede para mexer a mão."""
        return bool(self._amostras) and segundos - self._ultima_aceita >= self.paciencia
