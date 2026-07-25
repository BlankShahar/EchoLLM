#!/bin/bash

# Resolve immutable replay inputs from a completed result directory. This file
# is sourced by submission helpers and intentionally does not submit jobs.

: "${ARTIFACT_RESULTS_DIR:?Missing completed result directory}"
: "${EXPERIMENT_PATH:?Missing experiment artifact root}"
: "${PYTHON_BIN:?Missing Python interpreter}"

ARTIFACT_RESULTS_DIR=$(cd "$ARTIFACT_RESULTS_DIR" && pwd)

yaml_config_value() {
  "$PYTHON_BIN" -c \
    'import sys,yaml; payload=yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}; value=payload.get(sys.argv[2], {}).get(sys.argv[3]); print(value or "")' \
    "$1" "$2" "$3"
}

ARTIFACT_CONFIG="${ARTIFACT_CONFIG:-}"
if [[ -z "$ARTIFACT_CONFIG" ]]; then
  ARTIFACT_CONFIG=$(
    find "$ARTIFACT_RESULTS_DIR/tasks" -mindepth 2 -maxdepth 2 \
      -name config.json -type f -print 2>/dev/null |
      sort | head -n 1 || true
  )
fi
if [[ -z "$ARTIFACT_CONFIG" ]]; then
  for candidate in \
    incremental_config.yaml \
    supplemental_config.yaml \
    experiment_config.yaml \
    existing_config.yaml \
    baseline_config.yaml; do
    if [[ -f "$ARTIFACT_RESULTS_DIR/$candidate" ]]; then
      ARTIFACT_CONFIG="$ARTIFACT_RESULTS_DIR/$candidate"
      break
    fi
  done
fi

if [[ -z "$ARTIFACT_CONFIG" || ! -f "$ARTIFACT_CONFIG" ]]; then
  echo "Could not find an experiment config under $ARTIFACT_RESULTS_DIR" >&2
  echo "Set ARTIFACT_CONFIG to a task config from the completed run." >&2
  exit 3
fi

RECORDED_LLM_PATH="${RECORDED_LLM_PATH:-$(yaml_config_value "$ARTIFACT_CONFIG" llm recorded_path)}"
EMBEDDING_CACHE_PATH="${EMBEDDING_CACHE_PATH:-$(yaml_config_value "$ARTIFACT_CONFIG" embedding cache_path)}"

ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-}"
if [[ -z "$ARTIFACT_PREFIX" && -f "$ARTIFACT_RESULTS_DIR/merge_manifest.json" ]]; then
  ORIGINAL_RESULTS=$(
    "$PYTHON_BIN" -c \
      'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); print(payload.get("baseline_directory") or payload.get("existing_directory") or "")' \
      "$ARTIFACT_RESULTS_DIR/merge_manifest.json"
  )
  if [[ -n "$ORIGINAL_RESULTS" ]]; then
    ORIGINAL_GROUP=$(basename "$ORIGINAL_RESULTS")
    ARTIFACT_PREFIX="${ORIGINAL_GROUP%-*}"
  fi
fi
if [[ -z "$ARTIFACT_PREFIX" ]]; then
  RESULT_GROUP=$(basename "$ARTIFACT_RESULTS_DIR")
  ARTIFACT_PREFIX="${RESULT_GROUP%-*}"
fi

PREPARED_PAIRS_PATH="${PREPARED_PAIRS_PATH:-}"
if [[ -z "$PREPARED_PAIRS_PATH" ]]; then
  PREPARED_PAIRS_PATH=$(
    find "$EXPERIMENT_PATH/prepared" -maxdepth 1 -type f \
      -name "${ARTIFACT_PREFIX}-*.jsonl.gz" -printf '%T@ %p\n' 2>/dev/null |
      sort -nr | cut -d' ' -f2- | head -n 1 || true
  )
fi

if [[ ! -f "$RECORDED_LLM_PATH" ]]; then
  echo "Recorded LLM database not found: ${RECORDED_LLM_PATH:-<unset>}" >&2
  echo "Set RECORDED_LLM_PATH to the database used by the completed run." >&2
  exit 3
fi
if [[ ! -f "$EMBEDDING_CACHE_PATH" ]]; then
  echo "Embedding cache not found: ${EMBEDDING_CACHE_PATH:-<unset>}" >&2
  echo "Set EMBEDDING_CACHE_PATH to the cache used by the completed run." >&2
  exit 3
fi
if [[ ! -f "$PREPARED_PAIRS_PATH" ]]; then
  echo "Prepared trace not found: ${PREPARED_PAIRS_PATH:-<unset>}" >&2
  echo "Set PREPARED_PAIRS_PATH to the trace used by the completed run." >&2
  exit 3
fi

echo "Artifact config:          $ARTIFACT_CONFIG"
echo "Recorded LLM database:    $RECORDED_LLM_PATH"
echo "Prepared request trace:   $PREPARED_PAIRS_PATH"
echo "Embedding cache:          $EMBEDDING_CACHE_PATH"
