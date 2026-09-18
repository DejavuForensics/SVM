#!/usr/bin/env bash

if [ -d "env" ]; then
  echo "Directory env already exists"
else
  python -m venv env
fi