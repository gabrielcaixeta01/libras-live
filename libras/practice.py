"""Modo prática: o app pede uma letra, você sinaliza, ele confere.

O resto do projeto vai na direção da tradução — texto sai da sua mão. Aqui a
seta aponta ao contrário: a letra vem primeiro e você tem que produzi-la. É a
diferença entre ler e escrever, e é o que realmente treina datilologia.

Lógica pura, como o `stabilizer` e o `speller`: nada de relógio, câmera ou
desenho aqui dentro. O tempo entra por parâmetro, então os testes são
determinísticos e a sessão inteira roda sem webcam.
"""

from __future__ import annotations

import numpy as np


class Pratica:
    """Uma sessão de N rodadas, uma letra alvo por rodada.

    Errar não passa a rodada. É de propósito: pular no erro deixaria você
    treinar só o que já sabe, e o placar mediria sorte. A rodada só anda quando
    a letra sai certa — ou quando você desiste dela.
    """

    def __init__(
        self,
        letras: list[str],
        rodadas: int = 10,
        semente: int | None = None,
    ):
        if not letras:
            raise ValueError("é preciso pelo menos uma letra para praticar")
        if rodadas < 1:
            raise ValueError("rodadas deve ser >= 1")

        self.letras = list(letras)
        self.total = rodadas
        self._semente = semente
        self._rng = np.random.default_rng(semente)

        self._sequencia: list[str] = []
        self._indice = 0
        self._acertos = 0
        self._erros = 0
        self._puladas = 0
        self._tempos: list[float] = []

        self._sortear()

    def _sortear(self) -> None:
        """Sorteia a sequência evitando a mesma letra duas vezes seguidas.

        Repetir cansa e não ensina: você fica com a mão já na posição certa.
        """
        sequencia: list[str] = []
        for _ in range(self.total):
            opcoes = [l for l in self.letras if l != (sequencia[-1] if sequencia else None)]
            if not opcoes:  # alfabeto de uma letra só
                opcoes = self.letras
            sequencia.append(str(self._rng.choice(opcoes)))
        self._sequencia = sequencia

    # --- estado ---

    @property
    def alvo(self) -> str | None:
        """A letra pedida agora, ou None se a sessão acabou."""
        if self.concluida:
            return None
        return self._sequencia[self._indice]

    @property
    def concluida(self) -> bool:
        return self._indice >= self.total

    @property
    def rodada(self) -> int:
        """Rodada atual, contando de 1. Fica em `total` depois do fim."""
        return min(self._indice + 1, self.total)

    @property
    def acertos(self) -> int:
        return self._acertos

    @property
    def erros(self) -> int:
        return self._erros

    @property
    def puladas(self) -> int:
        return self._puladas

    @property
    def tempo_medio(self) -> float:
        """Segundos médios até acertar. Só conta rodadas acertadas."""
        return float(np.mean(self._tempos)) if self._tempos else 0.0

    @property
    def precisao(self) -> float:
        """Acertos sobre tentativas. Pular não conta como tentativa."""
        tentativas = self._acertos + self._erros
        return self._acertos / tentativas if tentativas else 0.0

    # --- interação ---

    def registrar(self, letra: str, segundos: float) -> bool:
        """Registra uma letra confirmada pelo estabilizador.

        Args:
            letra: a letra que o usuário sinalizou.
            segundos: tempo desde o início da rodada.

        Returns:
            True se era a letra pedida (e a rodada avançou).
        """
        if self.concluida:
            return False

        if letra != self.alvo:
            self._erros += 1
            return False

        self._acertos += 1
        self._tempos.append(segundos)
        self._indice += 1
        return True

    def pular(self, segundos: float = 0.0) -> None:
        """Desiste da rodada atual e vai para a próxima."""
        if self.concluida:
            return
        self._puladas += 1
        self._indice += 1

    def reiniciar(self) -> None:
        """Zera o placar e sorteia uma sequência nova."""
        self._indice = 0
        self._acertos = 0
        self._erros = 0
        self._puladas = 0
        self._tempos.clear()
        self._rng = np.random.default_rng(self._semente)
        self._sortear()

    def resumo(self) -> str:
        """Uma linha com o placar, para imprimir no fim."""
        partes = [f"{self._acertos}/{self.total} acertos"]
        if self._erros:
            partes.append(f"{self._erros} erros")
        if self._puladas:
            partes.append(f"{self._puladas} puladas")
        if self._tempos:
            partes.append(f"{self.tempo_medio:.1f}s por letra")
        return " | ".join(partes)
