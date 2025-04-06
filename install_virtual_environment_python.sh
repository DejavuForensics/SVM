#!/usr/bin/env bash

if [ $(id -u) -ne 0 ]
  then echo Please run this script as root or using sudo!
  exit
fi

apt update && apt install python3.11-venv