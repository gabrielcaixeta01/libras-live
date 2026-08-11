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

O modelo é **fechado nas letras que viu treinando**: dada qualquer mão, ele
responde a mais parecida entre elas, nunca "não sei". Abaixo de
`CONFIANCA_REJEICAO` a predição vira um `?` amarelo na tela e não entra no
texto, e a faixa de letras no topo mostra apagadas as que ele não conhece — o
app não finge saber o alfabeto inteiro.

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

Quando o MediaPipe não acha a mão numa imagem, ela não é descartada de cara: o
script reapresenta a mesma foto espelhada, ampliada, com contraste equalizado e
girada, e desfaz a transformação nos pontos quando alguma variante funciona.
São 377 imagens recuperadas assim, quase todas de mão fechada — **N sozinha vai
de 37 para 138 amostras**.

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

Compara Random Forest, MLP e SVM e salva o melhor, medindo por **macro-F1**: as
classes são desiguais, e acurácia global esconde justamente as letras raras
(errar N inteiro custa menos de 1% de acurácia). Cada classe do treino é
completada até 600 amostras com variações sintéticas — rotação, jitter e
profundidade sobre os landmarks. O aumento acontece dentro de cada fold, nunca
antes de separar treino e teste, senão a mesma mão cairia dos dois lados e a
métrica subiria sozinha.

O relatório completo (por classe, matriz de confusão, piores letras) fica em
`models/relatorio_treino.txt`.

**4. Rodar**

```bash
python -m libras.app
```

`ESC` sai · `BACKSPACE` apaga · `C` limpa · tirar a mão de cena por 1,5s insere
um espaço. Ao sair, o texto vai para a área de transferência.

**5. Praticar** (opcional)

```bash
python -m libras.app --praticar
```

Inverte o exercício: em vez de você soletrar o que quiser, o app pede uma letra
e confere se você acertou, cronometrando. Errar não passa a rodada — senão você
treinaria só o que já sabe. `P` pula a letra, `R` reinicia a sessão.

## Ajustes

Tudo em [`libras/config.py`](libras/config.py). Os três que importam:

- `TAMANHO_BUFFER` (15) — maior deixa mais estável e mais lento.
- `CONFIANCA_MINIMA` (0.70) — maior erra menos e engole mais letras.
- `CONFIANCA_REJEICAO` (0.55) — abaixo disso a predição vira `?` e é ignorada.

## Estado atual

Treinado só com a base pública, o modelo cobre **15 das 26 letras** (A B C D E I
L M N O R S U V W) e dá **99,6% de macro-F1** no conjunto de teste, que é
composto só de amostras reais.

Esse número mede o quão bem ele aprendeu *aquele dataset*, não o quanto vai
acertar na sua webcam — as condições são outras. Espere menos na prática, e
grave amostras suas das letras que estiverem falhando. As piores hoje são W e V
(98,3% e 98,4%), que o modelo troca entre si.

As 11 letras restantes (F G H J K P Q T X Y Z) não têm dado nenhum: aparecem
apagadas na faixa do topo e o modelo nunca as prevê. Sinalizar uma delas produz
um `?` ou a letra conhecida mais parecida — grave-as com `collect.py`.

## Testes

```bash
python -m pytest tests/ -q
```

96 testes cobrindo normalização, estabilização, soletração, aumento de dados,
recuperação de imagens, rejeição e modo prática — toda a lógica que pode dar
errado, sem precisar de câmera ou modelo treinado.

## Documentação

[Design da fase 1](docs/superpowers/specs/2026-08-11-libras-live-design.md) —
decisões, alternativas rejeitadas e o porquê.
