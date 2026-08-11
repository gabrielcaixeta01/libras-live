# libras-live

Reconhecimento do **alfabeto de Libras** em tempo real pela webcam. Você
sinaliza, ele vai formando o texto na tela — contínuo, sem apertar botão a cada
letra.

Feito como ferramenta de treino de datilologia.

## O que ele faz e o que ele não faz

Reconhece as **26 letras do alfabeto de Libras** (Língua Brasileira de Sinais —
não ASL) a partir da pose da mão, e monta palavras conforme você soletra.

Não traduz Libras como língua: frases, gramática espacial, expressão facial e
sinais compostos estão fora do escopo. Isso é problema de pesquisa em aberto, não
uma limitação a ser corrigida numa próxima versão.

Letras com movimento (H, J, K, X, Z) são reconhecidas pela pose característica,
não pela trajetória. Funciona para treinar, mas é uma simplificação — sinais com
movimento de verdade ficam para a fase 2.

## Como funciona

```
frame → 21 landmarks da mão → normalizar → classificar → estabilizar → texto
```

O classificador nunca vê a imagem. O MediaPipe extrai 21 pontos 3D da mão, esses
pontos são normalizados (pulso na origem, escala unitária) e só então
classificados. Por isso o modelo é indiferente a fundo, roupa, iluminação, cor de
pele e distância da câmera — o que também o deixa minúsculo (~1MB) e rápido
(30fps em CPU).

A **estabilização** é a peça que faz a diferença entre um demo e algo usável.
Classificar frame a frame produz texto tremido, porque uma fração dos frames
sempre erra. Uma letra só é confirmada quando domina 80% de uma janela de 15
frames com confiança média acima de 70%, e fica bloqueada até a mão mudar de
estado — senão segurar a letra emitiria `AAAAAAA`.

## Instalação

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_model.sh
```

## Uso

**1. Preparar a base pública** (~5 min, baixa 57MB)

```bash
python training/prepare_dataset.py
```

Usa o [Brazilian Sign Language Alphabet
Dataset](https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset),
4.411 imagens de Libras. Ele cobre 15 letras: A B C D E I L M N O R S U V W.

**2. Gravar as 11 letras que faltam** (~15 min)

```bash
python training/collect.py
```

Grava F G H J K P Q T X Y Z pela webcam, 200 amostras cada. Dá para parar no meio
e continuar depois — o script pula o que já gravou. Mexa a mão devagar enquanto
grava: gire o pulso, aproxime e afaste. Amostras idênticas ensinam o modelo a
reconhecer uma pose exata em vez da letra.

**3. Treinar**

```bash
python training/train.py
```

Compara Random Forest, MLP e SVM por validação cruzada e salva o melhor. No fim
imprime quais letras o modelo mais confunde — é onde vale gravar mais amostras.

**4. Rodar**

```bash
python -m libras.app
```

`ESC` sai · `BACKSPACE` apaga · `C` limpa · tirar a mão de cena por 1,5s insere
um espaço.

## Ajustes

Tudo em [`libras/config.py`](libras/config.py). Os dois que importam:

- `TAMANHO_BUFFER` (15) — maior deixa mais estável e mais lento.
- `CONFIANCA_MINIMA` (0.70) — maior erra menos e engole mais letras.

## Estado atual

Treinado só com a base pública (15 letras), o modelo dá **99,5%** no conjunto de
teste. Esse número mede o quão bem ele aprendeu *aquele dataset*, não o quanto
vai acertar na sua webcam — as condições são outras. Espere menos na prática, e
grave amostras suas das letras que estiverem falhando.

Duas letras merecem atenção: o MediaPipe detecta mão em só 24% das imagens de
**N** e 53% das de **M** na base pública — poses de mão fechada escondem os
dedos. Sobraram poucas amostras dessas letras, então elas são as primeiras
candidatas a receber gravações suas.

## Testes

```bash
python -m pytest tests/ -q
```

Cobrem normalização, estabilização e soletração — toda a lógica que pode dar
errado, sem precisar de câmera ou modelo treinado.

## Documentação

[Design da fase 1](docs/superpowers/specs/2026-08-11-libras-live-design.md) —
decisões, alternativas rejeitadas e o porquê.
