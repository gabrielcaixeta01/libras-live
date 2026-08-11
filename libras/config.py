"""Parâmetros ajustáveis.

Os valores de estabilização são os que você mais vai mexer no uso real:
buffer maior = mais estável e mais lento; limiar maior = menos erro e mais
letras engolidas.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

MODELO_MAOS = RAIZ / "models" / "hand_landmarker.task"
MODELO_POSE = RAIZ / "models" / "pose_landmarker.task"
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

# Uma amostra só é aceita se estiver a esta distância de todas as já gravadas.
# É o que impede 200 cópias do mesmo frame: sem isso a coleta termina em 8s e a
# nuvem da letra fica com raio ~0.12, contra ~0.64 das letras da base pública.
DISTANCIA_MINIMA_AMOSTRA = 0.12
PACIENCIA_COLETA = 1.5     # segundos sem aceitar nada antes de afrouxar o limiar
DECAIMENTO_COLETA = 0.8    # quanto o limiar encolhe a cada afrouxada
DISPERSAO_ALVO = 0.35      # raio de nuvem considerado saudável, para a UI


# =============================================================================
# Sinais — python -m libras.app --sinais (fase 2)
#
# Nada abaixo desta linha afeta o alfabeto. Ver a spec em
# docs/superpowers/specs/2026-08-11-sinais-dicionario-design.md
# =============================================================================

DIR_SINAIS = DIR_DADOS / "sinais"
DICIONARIO_SINAIS = DIR_SINAIS / "dicionario.npz"
PROTOTIPOS_USUARIO = DIR_SINAIS / "meus_prototipos.npz"
RELATORIO_SINAIS = RAIZ / "models" / "relatorio_sinais.txt"

# --- Segmentação (libras/sinais/segmenter.py) ---
# Velocidade em larguras de ombro por segundo — mesma invariante da
# normalização, então chegar perto da câmera não dispara sinal sozinho.
LIMIAR_MOVIMENTO = 0.35
FRAMES_PARA_INICIAR = 3      # tranco isolado não é consulta
SEGUNDOS_REPOUSO = 0.5       # mão parada por este tempo fecha o sinal
SEGUNDOS_MINIMO_SINAL = 0.4  # abaixo disso foi ruído; descarta em silêncio
SEGUNDOS_MAXIMO_SINAL = 4.0  # corta em vez de gravar para sempre

# --- Busca ---
CANDIDATOS_NA_TELA = 5   # é um dicionário, não um tradutor: cinco é resposta boa
BANDA_DTW = 0.2          # desvio máximo da diagonal no alinhamento temporal
SEGUNDOS_MOSTRANDO = 4.0  # quanto tempo o resultado fica na tela
