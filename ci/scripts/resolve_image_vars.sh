#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  echo "usage: $0 <deploy_env> [image_tag] [config_file]" >&2
  exit 1
fi

DEPLOY_ENV="$1"
IMAGE_TAG_INPUT="${2:-}"
CONFIG_FILE_INPUT="${3:-}"

if [[ -n "$CONFIG_FILE_INPUT" ]]; then
  CONFIG_FILE="$CONFIG_FILE_INPUT"
elif [[ -f ".docker.env" ]]; then
  CONFIG_FILE=".docker.env"
elif [[ -f "docker.env" ]]; then
  CONFIG_FILE="docker.env"
elif [[ -f ".docker.env.example" ]]; then
  CONFIG_FILE=".docker.env.example"
else
  CONFIG_FILE="docker.env.example"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

SERVICE_NAME_RESOLVED="${GH_SERVICE_NAME:-${SERVICE_NAME:-}}"
REGISTRY_HOST_RESOLVED="${GH_REGISTRY_HOST:-${REGISTRY_HOST:-}}"
REGISTRY_REPOSITORY_RESOLVED="${GH_REGISTRY_REPOSITORY:-${REGISTRY_REPOSITORY:-}}"
REGISTRY_PROJECT_RESOLVED="${GH_REGISTRY_PROJECT:-${REGISTRY_PROJECT:-}}"

if [[ -z "$SERVICE_NAME_RESOLVED" ]]; then
  echo "SERVICE_NAME is empty after resolving config and overrides" >&2
  exit 1
fi
if [[ -z "$REGISTRY_HOST_RESOLVED" ]]; then
  echo "REGISTRY_HOST is empty after resolving config and overrides" >&2
  exit 1
fi
if [[ -z "$REGISTRY_REPOSITORY_RESOLVED" ]]; then
  echo "REGISTRY_REPOSITORY is empty after resolving config and overrides" >&2
  exit 1
fi
if [[ -z "$REGISTRY_PROJECT_RESOLVED" ]]; then
  echo "REGISTRY_PROJECT is empty after resolving config and overrides" >&2
  exit 1
fi

IMAGE_TAG_RESOLVED="${IMAGE_TAG_INPUT:-$DEPLOY_ENV}"
IMAGE_REPOSITORY_RESOLVED="${REGISTRY_HOST_RESOLVED}/${REGISTRY_PROJECT_RESOLVED}/${REGISTRY_REPOSITORY_RESOLVED}/${SERVICE_NAME_RESOLVED}"
IMAGE_URI_RESOLVED="${IMAGE_REPOSITORY_RESOLVED}:${IMAGE_TAG_RESOLVED}"

cat <<EOF
DEPLOY_ENV=$DEPLOY_ENV
SERVICE_NAME=$SERVICE_NAME_RESOLVED
REGISTRY_HOST=$REGISTRY_HOST_RESOLVED
REGISTRY_PROJECT=$REGISTRY_PROJECT_RESOLVED
REGISTRY_REPOSITORY=$REGISTRY_REPOSITORY_RESOLVED
IMAGE_TAG=$IMAGE_TAG_RESOLVED
IMAGE_REPOSITORY=$IMAGE_REPOSITORY_RESOLVED
IMAGE_URI=$IMAGE_URI_RESOLVED
EOF
