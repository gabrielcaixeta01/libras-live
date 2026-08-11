import cv2
import numpy as np
import pytest

from libras.landmarks import NUM_PONTOS, Deteccao, normalizar
from libras.recovery import (
    VARIANTES,
    aplicar_afim,
    desespelhar,
    detectar_com_tentativas,
    girar,
)


def imagem_falsa(altura: int = 60, largura: int = 80) -> np.ndarray:
    """Retângulo com um quadrado claro fora do centro, para a rotação ter efeito."""
    imagem = np.zeros((altura, largura, 3), dtype=np.uint8)
    imagem[10:25, 15:35] = 200
    return imagem


def pontos_falsos(semente: int = 0) -> np.ndarray:
    """21 pontos em coordenadas normalizadas de imagem (0..1), como o MediaPipe."""
    rng = np.random.default_rng(semente)
    pontos = rng.uniform(0.2, 0.8, size=(NUM_PONTOS, 3)).astype(np.float32)
    pontos[:, 2] = rng.uniform(-0.1, 0.1, size=NUM_PONTOS)
    return pontos


# --- aplicar_afim ---


def test_afim_identidade_nao_move_nada():
    identidade = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    pontos = pontos_falsos()
    movidos = aplicar_afim(pontos, identidade, (60, 80), (60, 80))
    assert np.allclose(movidos, pontos, atol=1e-5)


def test_afim_preserva_a_profundidade():
    matriz = cv2.getRotationMatrix2D((40.0, 30.0), 25.0, 1.0)
    pontos = pontos_falsos()
    movidos = aplicar_afim(pontos, matriz, (60, 80), (60, 80))
    assert np.allclose(movidos[:, 2], pontos[:, 2], atol=1e-6)


# --- girar ---


def test_giro_nulo_devolve_a_mesma_imagem():
    imagem = imagem_falsa()
    girada, _ = girar(imagem, 0.0)
    assert girada.shape == imagem.shape
    assert np.array_equal(girada, imagem)


def test_giro_de_90_troca_as_dimensoes():
    girada, _ = girar(imagem_falsa(60, 80), 90.0)
    assert girada.shape[:2] == (80, 60)


def test_giro_expande_a_tela_para_nao_cortar():
    """A 45° a imagem não cabe no retângulo original."""
    girada, _ = girar(imagem_falsa(60, 80), 45.0)
    assert girada.shape[0] > 60 and girada.shape[1] > 80


@pytest.mark.parametrize("graus", [15.0, -20.0, 40.0, -45.0])
def test_pontos_voltam_ao_lugar_depois_de_desgirar(graus):
    """O ponto detectado na imagem girada tem que voltar à coordenada original."""
    imagem = imagem_falsa()
    girada, matriz = girar(imagem, graus)
    forma_orig = imagem.shape[:2]
    forma_girada = girada.shape[:2]

    pontos = pontos_falsos()
    na_girada = aplicar_afim(pontos, matriz, forma_orig, forma_girada)
    de_volta = aplicar_afim(
        na_girada, cv2.invertAffineTransform(matriz), forma_girada, forma_orig
    )

    assert np.allclose(de_volta, pontos, atol=1e-4)


# --- desespelhar ---


def test_desespelhar_inverte_x_e_preserva_o_resto():
    pontos = pontos_falsos()
    espelhados = desespelhar(pontos)
    assert np.allclose(espelhados[:, 0], 1.0 - pontos[:, 0], atol=1e-6)
    assert np.allclose(espelhados[:, 1:], pontos[:, 1:], atol=1e-6)


def test_desespelhar_e_involucao():
    pontos = pontos_falsos()
    assert np.allclose(desespelhar(desespelhar(pontos)), pontos, atol=1e-6)


# --- cascata de variantes ---


class DetectorFalso:
    """Detector que só acha mão nas variantes cujo nome está em `aceita`."""

    def __init__(self, aceita: set[str], mao_esquerda: bool = False):
        self.aceita = aceita
        self.mao_esquerda = mao_esquerda
        self.chamadas: list[tuple[int, int]] = []
        self._nomes = [v.nome for v in VARIANTES]
        self._proxima = 0

    def detectar(self, imagem):
        nome = self._nomes[self._proxima]
        self._proxima += 1
        self.chamadas.append(nome)

        if nome not in self.aceita:
            return None

        pontos = pontos_falsos(1)
        return Deteccao(
            pontos=pontos,
            vetor=normalizar(pontos, self.mao_esquerda),
            mao_esquerda=self.mao_esquerda,
        )


def test_primeira_variante_e_a_imagem_original():
    assert VARIANTES[0].nome == "original"


def test_nomes_das_variantes_sao_unicos():
    nomes = [v.nome for v in VARIANTES]
    assert len(nomes) == len(set(nomes))


def test_para_na_primeira_variante_que_funciona():
    detector = DetectorFalso(aceita={"original"})
    deteccao, nome = detectar_com_tentativas(detector, imagem_falsa())

    assert deteccao is not None
    assert nome == "original"
    assert detector.chamadas == ["original"]


def test_tenta_as_seguintes_quando_a_original_falha():
    segunda = VARIANTES[1].nome
    detector = DetectorFalso(aceita={segunda})
    deteccao, nome = detectar_com_tentativas(detector, imagem_falsa())

    assert deteccao is not None
    assert nome == segunda
    assert detector.chamadas == ["original", segunda]


def test_desiste_depois_de_todas_as_variantes():
    detector = DetectorFalso(aceita=set())
    deteccao, nome = detectar_com_tentativas(detector, imagem_falsa())

    assert deteccao is None
    assert nome is None
    assert len(detector.chamadas) == len(VARIANTES)


def test_variante_espelhada_inverte_a_lateralidade():
    """Numa imagem espelhada o MediaPipe reporta a mão trocada."""
    detector = DetectorFalso(aceita={"espelhada"}, mao_esquerda=False)
    deteccao, nome = detectar_com_tentativas(detector, imagem_falsa())

    assert nome == "espelhada"
    assert deteccao.mao_esquerda is True


def test_deteccao_recuperada_sai_normalizada():
    detector = DetectorFalso(aceita={VARIANTES[-1].nome})
    deteccao, _ = detectar_com_tentativas(detector, imagem_falsa())

    pontos = deteccao.vetor.reshape(NUM_PONTOS, 3)
    assert np.allclose(pontos[0], 0, atol=1e-5)
    assert np.isclose(np.max(np.linalg.norm(pontos, axis=1)), 1.0, atol=1e-5)
