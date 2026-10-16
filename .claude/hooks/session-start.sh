#!/bin/bash
# Installs everything the reel-editor skill needs in a fresh cloud container.
set -uo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

SKILL="$CLAUDE_PROJECT_DIR/.claude/skills/reel-editor"

# Python packages
pip install -q --root-user-action=ignore pillow numpy arabic-reshaper python-bidi faster-whisper onnxruntime fonttools \
  || echo "reel-editor: pip install failed" >&2

# Arabic fonts (from npm — Google Fonts' GitHub raw URLs are blocked here)
mkdir -p ~/.fonts
if [ ! -f ~/.fonts/NotoKufiArabic-Bold.ttf ] || [ ! -f ~/.fonts/Cairo-Bold.ttf ]; then
  tmp=$(mktemp -d)
  (cd "$tmp" && npm pack -q @expo-google-fonts/noto-kufi-arabic @expo-google-fonts/cairo >/dev/null 2>&1 \
    && for t in *.tgz; do tar xzf "$t" --one-top-level; done \
    && cp */package/700Bold/NotoKufiArabic_700Bold.ttf ~/.fonts/NotoKufiArabic-Bold.ttf \
    && cp */package/700Bold/Cairo_700Bold.ttf ~/.fonts/Cairo-Bold.ttf) \
    || echo "reel-editor: font install failed" >&2
  rm -rf "$tmp"
fi

# Person-segmentation model for behind-the-speaker effects
python3 -c "import sys; sys.path.insert(0, '$SKILL/scripts'); import matte; matte.ensure_model()" \
  || echo "reel-editor: matte model download failed" >&2

# Whisper model (needs huggingface.co allowed in the environment's network settings)
timeout 300 python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" >/dev/null 2>&1 \
  || echo "reel-editor: Whisper model not downloaded (is huggingface.co allowed?)" >&2

exit 0
