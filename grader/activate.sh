#!/bin/bash
if [ ! -d "venv" ]; then
  echo "❌ Виртуальное окружение 'venv' не найдено!"
  exit 1
fi

source venv/bin/activate
export WANDB_MODE="offline"
echo "✅ venv активирован, WANDB_MODE=offline"
