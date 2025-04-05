#/bin/bash

if [ $(id -u) -ne 0 ]
  then echo Please run this script as root or using sudo!
  exit
fi

apt install python3.11-venv
python -m venv venv
source venv/bin/activate
