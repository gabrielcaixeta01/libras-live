"""Decide quando uma predição vira uma letra de verdade.

Classificar frame a frame produz lixo: uma fração dos frames sempre erra, e o
texto sai tremido (`AAAABAAAA`). Além disso, segurar a mão parada numa letra
emitiria a mesma letra a 30 por segundo.

Este módulo resolve os dois problemas e é lógica pura — dá para testar sem
câmera, sem modelo e sem relógio.
"""

from __future__ import annotations

from collections import Counter, deque

from .. import config


class Estabilizador:
    """Converte um fluxo ruidoso de predições em letras confirmadas.

    Uma letra é confirmada quando domina o buffer recente **e** vem com
    confiança média suficiente. Depois de confirmada, fica bloqueada até a mão
    mudar de estado — é o que impede a letra segurada de repetir.
    """

    def __init__(
        self,
        tamanho_buffer: int = config.TAMANHO_BUFFER,
        dominancia_minima: float = config.DOMINANCIA_MINIMA,
        confianca_minima: float = config.CONFIANCA_MINIMA,
    ):
        if tamanho_buffer < 1:
            raise ValueError("tamanho_buffer deve ser >= 1")

        self.tamanho_buffer = tamanho_buffer
        self.dominancia_minima = dominancia_minima
        self.confianca_minima = confianca_minima

        self._buffer: deque[tuple[str | None, float]] = deque(maxlen=tamanho_buffer)
        self._bloqueada: str | None = None

    def atualizar(self, letra: str | None, confianca: float = 0.0) -> str | None:
        """Registra a predição de um frame.

        Args:
            letra: classe predita, ou None se não havia mão no frame.
            confianca: probabilidade da classe predita, entre 0 e 1.

        Returns:
            A letra, no exato frame em que ela é confirmada. None nos demais.
        """
        self._buffer.append((letra, confianca))

        if len(self._buffer) < self.tamanho_buffer:
            return None

        candidata, ocorrencias = Counter(l for l, _ in self._buffer).most_common(1)[0]

        # Perdeu a dominância: a mão mudou de estado, libera o bloqueio.
        if candidata != self._bloqueada:
            self._bloqueada = None

        if candidata is None:
            return None

        if ocorrencias / self.tamanho_buffer < self.dominancia_minima:
            return None

        confiancas = [c for l, c in self._buffer if l == candidata]
        if sum(confiancas) / len(confiancas) < self.confianca_minima:
            return None

        if candidata == self._bloqueada:
            return None

        self._bloqueada = candidata
        return candidata

    def reiniciar(self) -> None:
        self._buffer.clear()
        self._bloqueada = None

    @property
    def preenchimento(self) -> float:
        """Fração do buffer já preenchida — a UI usa para desenhar progresso."""
        return len(self._buffer) / self.tamanho_buffer
