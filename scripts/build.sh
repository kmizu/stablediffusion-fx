#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
exec sbt -batch ui/assembly
