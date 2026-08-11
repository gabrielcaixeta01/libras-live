import numpy as np
import pytest

from libras.alfabeto.classifier import Classificador, ModeloAusente, Predicao


class ModeloFalso:
    """Devolve sempre as mesmas probabilidades, na ordem das letras."""

    def __init__(self, probabilidades):
        self.probabilidades = np.array([probabilidades], dtype=np.float64)

    def predict_proba(self, X):
        return np.repeat(self.probabilidades, len(X), axis=0)


def classificador(probabilidades, letras="ABC", limiar=0.55) -> Classificador:
    return Classificador.de_pacote(
        {
            "modelo": ModeloFalso(probabilidades),
            "letras": list(letras),
            "acuracia": 0.9,
            "macro_f1": 0.8,
            "nome": "falso",
        },
        limiar_rejeicao=limiar,
    )


VETOR = np.zeros(63, dtype=np.float32)


def test_erro_amigavel_quando_o_modelo_nao_existe(tmp_path):
    with pytest.raises(ModeloAusente):
        Classificador(caminho=tmp_path / "nao_existe.joblib")


def test_prediz_a_classe_de_maior_probabilidade():
    predicao = classificador([0.1, 0.7, 0.2]).prever(VETOR)
    assert predicao.letra == "B"
    assert predicao.confianca == pytest.approx(0.7)


def test_aceita_acima_do_limiar():
    predicao = classificador([0.1, 0.7, 0.2], limiar=0.55).prever(VETOR)
    assert predicao.aceita
    assert predicao.rotulo == "B"


def test_rejeita_abaixo_do_limiar():
    """Uma letra que o modelo não conhece cai em empate entre as que ele conhece."""
    predicao = classificador([0.34, 0.33, 0.33], limiar=0.55).prever(VETOR)
    assert not predicao.aceita
    assert predicao.rotulo == "?"
    assert predicao.letra == "A"  # a letra continua disponível para depuração


def test_limiar_exato_e_aceito():
    predicao = classificador([0.55, 0.25, 0.20], limiar=0.55).prever(VETOR)
    assert predicao.aceita


def test_letras_ausentes_sao_o_resto_do_alfabeto():
    modelo = classificador([1.0, 0.0, 0.0], letras="ABC")
    assert "Z" in modelo.letras_ausentes
    assert "A" not in modelo.letras_ausentes
    assert len(modelo.letras) + len(modelo.letras_ausentes) == 26


def test_guarda_as_metricas_do_treino():
    modelo = classificador([1.0, 0.0, 0.0])
    assert modelo.macro_f1 == pytest.approx(0.8)
    assert modelo.acuracia == pytest.approx(0.9)


def test_pacote_antigo_sem_macro_f1_nao_quebra():
    modelo = Classificador.de_pacote(
        {"modelo": ModeloFalso([1.0]), "letras": ["A"], "acuracia": 0.5}
    )
    assert modelo.macro_f1 == 0.0
    assert modelo.nome == "desconhecido"


def test_predicao_e_imutavel():
    predicao = Predicao("A", 0.9, True)
    with pytest.raises(Exception):
        predicao.letra = "B"
