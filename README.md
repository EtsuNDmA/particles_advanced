# particles_advanced
Моделирование движения частиц в поле морских течений

## Установка

1. Установить [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Добавть канал conda-forge
    ```bash
    conda config --add channels conda-forge
   ```
3. Создать новое окружение
   ```bash
   conda create -n your-env-name python=3.7 --file requirements.txt
   ```
4. Активировать окружение
   ```bash
   conda activate your-env-name
   ```

## Запуск

```bash
python main.py
```

### Другое количество итераций

```bash
python main.py --num-iter=100
```

### Другое начальное расположение частиц

В качестве seed любое целое число
```bash
python main.py --seed=53
```

### Список всех опций

```bash
python main.py --help
```
