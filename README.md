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

**2. Gravar as suas amostras**

```bash
python training/collect.py            # tudo que ainda falta
python training/collect.py --revisar  # diagnóstico do que já foi gravado
```

Onze letras (F G H J K P Q T X Y Z) não existem na base pública e **só** podem
vir daqui. As outras quinze existem, mas como fotos de estúdio de mãos de outras
pessoas — e isso desequilibra o reconhecimento: uma letra com 200 amostras da
sua mão ganha de uma letra com 150 amostras de mãos estranhas, mesmo quando a
forma está certa. Foi assim que o L (só estúdio) começou a sair como G (só
webcam). Vale gravar as 26.

Nada de imagem sai daqui: cada frame vira 21 pontos e é descartado. O arquivo
salvo tem só os 63 floats por amostra.

**Amostra repetida não conta.** Uma amostra só entra se estiver a pelo menos
`DISTANCIA_MINIMA_AMOSTRA` de todas as que já entraram, então mexer a mão não é
conselho — é o que faz a barra andar. Gire o pulso, aproxime e afaste, incline.
Se a barra travar, a tela avisa.

Isso existe porque a primeira coleta não tinha o filtro: 200 amostras saíam em
oito segundos e a nuvem da letra ficava com raio 0,12, contra 0,64 das letras de
estúdio. O modelo decorava a pose exata em vez da letra. O `--revisar` mostra o
raio de cada gravação e aponta quais vale regravar.

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
- `DISTANCIA_MINIMA_AMOSTRA` (0.12) — quão diferentes as amostras da coleta
  precisam ser entre si.

## Estado atual

O modelo cobre **as 26 letras** e dá **99,6% de macro-F1** no conjunto de teste.

**Esse número não descreve o uso real, e a distância entre os dois é o assunto
em aberto do projeto.** O teste é feito na mesma distribuição do treino: quinze
letras vêm de fotos de estúdio, onze vêm de uma sessão de webcam. No uso, todo
frame vem da sua webcam — as quinze letras de estúdio estão fora da
distribuição, e é lá que aparecem as confusões que a métrica não vê (L saindo
como G, por exemplo).

Duas medidas expõem isso melhor que o macro-F1, e ambas saem de
`collect.py --revisar` e do relatório de treino:

- **Raio da nuvem de cada letra.** As gravações da primeira sessão ficaram entre
  0,12 e 0,26 em F G K Q T — pose decorada, não letra aprendida. O filtro de
  diversidade da coleta existe para isso; essas letras merecem `--refazer`.
- **Letras de movimento.** J tem raio 1,43 e invade M, C, A e L: as amostras não
  são uma pose, são uma trajetória inteira achatada num rótulo só. Tratar H J K
  X Z como pose é a simplificação declarada aqui em cima, e é ela que está
  chegando ao limite.

O que falta para ter um número honesto: uma sessão de gravação separada, em
outro dia e outra luz, usada só como teste e nunca no treino.

## Testes

```bash
python -m pytest tests/ -q
```

118 testes cobrindo normalização, estabilização, soletração, aumento de dados,
recuperação de imagens, rejeição, diversidade da coleta e modo prática — toda a
lógica que pode dar errado, sem precisar de câmera ou modelo treinado.

## Documentação

[Design da fase 1](docs/superpowers/specs/2026-08-11-libras-live-design.md) —
decisões, alternativas rejeitadas e o porquê.
