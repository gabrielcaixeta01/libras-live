"""As primitivas do MediaPipe — a parte delas que não precisa de modelo.

Carregar um landmarker de verdade exige o arquivo `.task` baixado, então isso
fica de fora, como toda a camada fina. O que dá para testar sem hardware é
exatamente onde os três detectores erravam igual: a conferência do modelo, a
montagem do array e a lateralidade espelhada.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from libras import mediapipe_io


def marco(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def lateralidade(nome: str) -> list:
    return [[SimpleNamespace(category_name=nome)]]


# --- conferência do modelo ---


def test_modelo_ausente_diz_como_resolver(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model.sh"):
        mediapipe_io.opcoes_base(tmp_path / "nao_existe.task")


def test_modelo_presente_passa(tmp_path):
    caminho = tmp_path / "modelo.task"
    caminho.write_bytes(b"")
    assert mediapipe_io.opcoes_base(caminho) is not None


# --- montagem dos pontos ---


def test_marcos_viram_array_de_tres_colunas():
    p = mediapipe_io.pontos([marco(0.1, 0.2, 0.3), marco(0.4, 0.5, 0.6)])
    assert p.shape == (2, 3)
    assert p.dtype == np.float32
    assert np.allclose(p[0], [0.1, 0.2, 0.3])


def test_lista_vazia_de_marcos():
    assert mediapipe_io.pontos([]).size == 0


# --- lateralidade ---


def test_le_a_mao_reportada():
    assert mediapipe_io.e_mao_esquerda(lateralidade("Left")) is True
    assert mediapipe_io.e_mao_esquerda(lateralidade("Right")) is False


def test_video_espelhado_inverte_o_lado():
    """O detector já corrige o espelho da câmera; se o app espelha de novo, a
    correção dele fica invertida. Sem isto, sinal assimétrico sai ao contrário."""
    assert mediapipe_io.e_mao_esquerda(lateralidade("Left"), video_espelhado=True) is (
        False
    )
    assert mediapipe_io.e_mao_esquerda(lateralidade("Right"), video_espelhado=True) is (
        True
    )


def test_segunda_mao_pelo_indice():
    duas = lateralidade("Left") + lateralidade("Right")
    assert mediapipe_io.e_mao_esquerda(duas, indice=1) is False


@pytest.mark.parametrize("entrada", [None, [], [[]]])
def test_sem_lateralidade_nao_quebra(entrada):
    assert mediapipe_io.e_mao_esquerda(entrada) is False


def test_indice_fora_da_faixa_nao_quebra():
    assert mediapipe_io.e_mao_esquerda(lateralidade("Left"), indice=7) is False


# --- ciclo de vida ---


class Falso:
    def __init__(self):
        self.fechado = False

    def close(self):
        self.fechado = True


class DetectorFalso(mediapipe_io.Landmarker):
    def __init__(self):
        self.a, self.b = Falso(), Falso()
        self._recursos = (self.a, self.b)


def test_sair_do_contexto_fecha_todos_os_recursos():
    with DetectorFalso() as d:
        assert not d.a.fechado
    assert d.a.fechado and d.b.fechado


def test_fecha_mesmo_com_excecao():
    d = DetectorFalso()
    with pytest.raises(RuntimeError):
        with d:
            raise RuntimeError("erro no meio do loop")
    assert d.a.fechado
