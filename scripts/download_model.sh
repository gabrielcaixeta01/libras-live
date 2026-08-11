#!/usr/bin/env bash
# Baixa o modelo de detecção de mãos do MediaPipe (~7.8MB).
# Não versionamos o arquivo no git por causa do tamanho.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$DIR/models/hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "$DIR/models"

if [ -f "$DESTINO" ]; then
  echo "Modelo já existe em $DESTINO"
  exit 0
fi

echo "Baixando modelo do MediaPipe..."
curl -fL --progress-bar -o "$DESTINO" "$URL"
echo "Salvo em $DESTINO"
