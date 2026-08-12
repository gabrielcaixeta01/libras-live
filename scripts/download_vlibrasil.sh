#!/usr/bin/env bash
# Baixa o V-LIBRASIL (~10,8 GB) do espelho público no Kaggle.
#
# O site oficial (https://libras.cin.ufpe.br) responde 502 desde antes desta
# escrita — o proxy da UFPE está de pé, o backend não. O espelho no Kaggle
# serve o bundle completo sem exigir login: a rota de download devolve 302
# para uma URL assinada do Google Cloud Storage.
#
# Licença do dataset: CC BY-NC-ND 4.0 (uso não comercial).
#
# O download é retomável: interromper e rodar de novo continua de onde parou.

set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$RAIZ/data/raw"
ZIP="$DESTINO/v-librasil.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/davimedio01/v-librasil"
TAMANHO_ESPERADO=10803251619

mkdir -p "$DESTINO"

echo "==> baixando V-LIBRASIL para $ZIP"
echo "    ~10,8 GB — retomável, pode interromper"

# --retry cobre queda de rede; -C - retoma de onde parou.
curl -L --fail --retry 10 --retry-delay 5 --retry-all-errors \
     -C - -o "$ZIP" "$URL"

TAMANHO_REAL=$(stat -f%z "$ZIP" 2>/dev/null || stat -c%s "$ZIP")
if [ "$TAMANHO_REAL" -ne "$TAMANHO_ESPERADO" ]; then
    echo "!! tamanho inesperado: $TAMANHO_REAL (esperado $TAMANHO_ESPERADO)" >&2
    echo "   rode de novo para retomar o download" >&2
    exit 1
fi

echo "==> download completo, verificando o zip"
unzip -tq "$ZIP"

# A extração é feita em Python, não com `unzip`: o pacote traz os nomes em UTF-8
# e o unzip do macOS os corrompe ("Abençoar" vira "Aben+?oar"), abortando com um
# falso "disk full". O script também reorganiza o layout plano do espelho em uma
# pasta por articulador, que é o que `libras.catalogo` lê.
echo "==> extraindo e organizando por articulador"
PYTHON="$RAIZ/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" "$RAIZ/scripts/organizar_vlibrasil.py" \
    --zip "$ZIP" --destino "$DESTINO/v-librasil"

echo "==> pronto"
