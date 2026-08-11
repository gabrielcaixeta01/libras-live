"""Parâmetros ajustáveis.

Os valores de estabilização são os que você mais vai mexer no uso real:
buffer maior = mais estável e mais lento; limiar maior = menos erro e mais
letras engolidas.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

MODELO_MAOS = RAIZ / "models" / "hand_landmarker.task"
MODELO_CLASSIFICADOR = RAIZ / "models" / "classifier.joblib"
RELATORIO_TREINO = RAIZ / "models" / "relatorio_treino.txt"
DIR_DADOS = RAIZ / "data"

# --- Câmera ---
INDICE_CAMERA = 0
LARGURA_CAMERA = 1280
ALTURA_CAMERA = 720
ESPELHAR_VIDEO = True  # efeito espelho: você se move para o lado que espera

# --- Estabilização (D4 da spec) ---
TAMANHO_BUFFER = 15        # frames considerados; ~0,5s a 30fps
DOMINANCIA_MINIMA = 0.80   # fração do buffer que a letra precisa ocupar
CONFIANCA_MINIMA = 0.70    # confiança média mínima da classe dominante
SEGUNDOS_PARA_ESPACO = 1.5  # mão ausente por este tempo emite um espaço

# --- Rejeição ---
# O classificador escolhe sempre a melhor entre as letras que conhece — se você
# sinalizar uma letra que ele nunca viu, ele responde a mais parecida, com
# confiança alta. Abaixo deste limiar a predição vira "?" e não conta.
CONFIANCA_REJEICAO = 0.55

# --- Modo prática (python -m libras.app --praticar) ---
RODADAS_PRATICA = 10

# --- Aumento de dados (training/train.py) ---
# Limites modestos de propósito: o aumento tem que gerar a mesma letra vista de
# outro jeito. Rotação demais transforma um M num W.
AUG_ROTACAO_GRAUS = 12.0     # rotação máxima por eixo, em graus
AUG_RUIDO = 0.02             # sigma do jitter, em frações do tamanho da mão
AUG_PROFUNDIDADE = 0.30      # variação máxima do eixo z (±30%)
AUG_ALVO_POR_CLASSE = 600    # amostras por letra depois de equilibrar

# --- Alfabeto ---
LETRAS_NO_DATASET = list("ABCDEILMNORSUVW")
LETRAS_A_COLETAR = list("FGHJKPQTXYZ")
ALFABETO = sorted(LETRAS_NO_DATASET + LETRAS_A_COLETAR)

# --- Coleta (training/collect.py) ---
AMOSTRAS_POR_LETRA = 200
SEGUNDOS_PREPARACAO = 3  # contagem regressiva antes de começar a gravar
