#!/usr/bin/env bash

if [ $(id -u) -ne 0 ]
  then echo Please run this script as root or using sudo!
  exit
fi

apt install update && apt install python3-virtualenv

if [ -d "env" ]; then
  echo "Directory env already exists"
else
  python -m venv env
  source env/bin/activate
fi