#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"/..

pdm export --without-hashes -o requirements.txt
