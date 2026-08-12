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

# O dicionário do encoder: mesmos protótipos, guardados como embedding. Quando
# ele existe, o app prefere ele — ver `app_sinais.carregar_busca`.
DICIONARIO_EMBEDDINGS = DIR_SINAIS / "dicionario_embeddings.npz"
PROTOTIPOS_USUARIO_EMBEDDINGS = DIR_SINAIS / "meus_prototipos_embeddings.npz"
ENCODER_SINAIS = RAIZ / "models" / "encoder_sinais.pt"
RELATORIO_ENCODER = RAIZ / "models" / "relatorio_encoder.txt"

# As distâncias medidas nos rodízios, com "a resposta estava na lista?" ao lado.
# É o que permite escolher outro limiar de rejeição sem repetir os três treinos.
CONFIANCAS_REJEICAO = RAIZ / "models" / "confiancas_rejeicao.npz"

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

# --- Rejeição nos sinais ---
# O índice tem 163 sinais e a Libras tem milhares: quase tudo que você pode
# sinalizar está fora dele. Sem limiar a busca responde assim mesmo, com os cinco
# mais parecidos e a mesma cara de resposta — a versão em sinais do problema que
# CONFIANCA_REJEICAO resolve no alfabeto.
#
# Calibrado por training/train_sinais.py --treino-nucleo: o quantil 95% das
# distâncias que acertaram, medido nos rodízios junto com 307 gravações de
# sinais fora do núcleo. Ver models/relatorio_encoder.txt.
#
# **O que ele compra é pouco, e o relatório mostra isso.** As duas distribuições
# se sobrepõem: preservando 95% dos acertos, o corte recusa só 8,5% das consultas
# que não tinham resposta. Custa ~1,8 ponto de recall@5 (37,3% → 35,5%) e em
# troca o app para de inventar nos piores casos. A distância do cosseno não é
# confiança, e enquanto ela for o único sinal disponível o limiar fica assim.
REJEICAO_COSSENO: float | None = 0.4723

# A baseline DTW não tem limiar calibrado: a distância dela não tem a mesma
# escala e o número que vale é o do encoder, que é o que o app usa. None desliga
# a rejeição em vez de inventar um corte.
REJEICAO_DTW: float | None = None


def limiar_de_rejeicao(metrica: str) -> float | None:
    """A distância acima da qual o app diz "não reconheci". None não rejeita nada."""
    return {"cosseno": REJEICAO_COSSENO, "dtw": REJEICAO_DTW}.get(metrica)

# --- Encoder neural (training/train_sinais.py) ---
# O número que o encoder existe para bater, medido pela baseline DTW no mesmo
# leave-one-articulator-out e registrado em models/relatorio_sinais.txt. Está
# aqui para que o relatório do encoder possa dar veredito sozinho.
BASELINE_DTW_RECALL5 = 0.075

# O mesmo número, no vocabulário núcleo — que é o que o app indexa por padrão.
# Medido com `python training/eval_sinais.py --nucleo` sobre 490 consultas.
# São 2,9x o vocabulário inteiro sem trocar nada na busca: só há 163 sinais para
# confundir em vez de 1.363.
BASELINE_DTW_NUCLEO_RECALL5 = 0.214

DIMENSAO_EMBEDDING = 256   # o vetor que substitui a sequência de 32×147
ENC_OCULTO = 256           # unidades por direção da GRU
ENC_CAMADAS = 2
ENC_DROPOUT = 0.3          # 2 gravações por classe: sem isto ele decora
ENC_EPOCAS = 60
ENC_LOTE = 64
ENC_TAXA = 1e-3            # AdamW com cosseno até quase zero
ENC_DECAIMENTO = 1e-4

# ArcFace: margem angular sobre 1.363 classes de 2 exemplos. A margem sobe de 0
# até este valor no primeiro terço do treino — começar já com ela põe a perda
# num platô do qual ela não sai.
ARCFACE_MARGEM = 0.30
ARCFACE_ESCALA = 32.0

# Aumento de dados dos sinais. Modesto pelo mesmo motivo do alfabeto: precisa
# gerar o mesmo sinal visto de outro jeito, não outro sinal. Sem espelhamento —
# lado é fonema aqui, e trocar a mão trocaria a palavra.
ENC_AUG_ROTACAO_GRAUS = 8.0   # giro em torno do eixo vertical e do da câmera
ENC_AUG_ESCALA = 0.10         # ±10% no tamanho aparente
ENC_AUG_RUIDO = 0.015         # sigma, em larguras de ombro
ENC_AUG_TEMPO = 0.20          # deformação do ritmo, mantendo começo e fim

# Vocabulário aberto: sinais retirados do treino do encoder que entram só como
# protótipo. É o que mede a promessa "adiciono um sinal novo com uma gravação".
VOCABULARIO_ABERTO = 200
