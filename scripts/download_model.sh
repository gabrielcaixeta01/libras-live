#!/usr/bin/env bash
# Baixa os modelos prontos do MediaPipe. Não são versionados no git por causa
# do tamanho, e são regeráveis por este script.
#
#   hand_landmarker  (~7.8MB)  alfabeto e sinais — 21 pontos por mão
#   pose_landmarker  (~9.0MB)  só sinais — 33 pontos do corpo, a âncora que
#                              permite saber *onde* no corpo o sinal acontece
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="https://storage.googleapis.com/mediapipe-models"

mkdir -p "$DIR/models"

baixar() {
  local destino="$DIR/models/$1"
  local url="$2"

  if [ -f "$destino" ]; then
    echo "já existe: $1"
    return 0
  fi

  echo "baixando $1..."
  curl -fL --progress-bar -o "$destino" "$url"
  echo "salvo em $destino"
}

baixar hand_landmarker.task \
  "$BASE/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

baixar pose_landmarker.task \
  "$BASE/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
