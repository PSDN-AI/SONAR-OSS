#!/bin/bash
# Download Common Voice Spontaneous Speech 1.0 - English from Mozilla Data Collective.
# Requires a (free) Mozilla Data Collective account API key.

OUTPUT_DIR="psdn_sonar/benchmarks/english/datasets/commonvoice"
mkdir -p "$OUTPUT_DIR"

DATASET_ID="cmihqzerk023co20749miafhq"
API_KEY="${MOZILLA_API_KEY:?Set MOZILLA_API_KEY env var}"

echo "Requesting presigned download URL..."
RESPONSE=$(curl -s -X POST "https://datacollective.mozillafoundation.org/api/datasets/${DATASET_ID}/download" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json")

DOWNLOAD_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('downloadUrl',''))" 2>/dev/null)

if [ -z "$DOWNLOAD_URL" ]; then
  echo "ERROR: Could not get download URL. Response:"
  echo "$RESPONSE"
  exit 1
fi

echo "Downloading Common Voice English to ${OUTPUT_DIR}..."
curl -L -o "${OUTPUT_DIR}/Common_Voice_English_1.0.tar.gz" "$DOWNLOAD_URL"

echo ""
echo "Download complete: ${OUTPUT_DIR}/Common_Voice_English_1.0.tar.gz"
echo ""
echo "Next steps:"
echo "  1. Extract: cd ${OUTPUT_DIR} && tar -xzf Common_Voice_English_1.0.tar.gz"
echo "  2. Convert to TSV for psdn-sonar (audio_path + transcription columns)"
