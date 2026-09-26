# Guia do Projeto Copo

## Ideia

O Projeto Copo é uma aplicação web que recebe uma foto de um copo e classifica a **aparência da água** como `Limpo` ou `Sujo`. A pessoa marca na foto a região que contém água; a classificação usa as cores dos pixels dessa região.

O resultado é apenas visual. Água transparente pode conter contaminantes invisíveis, e água com aparência diferente pode receber um rótulo incorreto. O sistema **não verifica potabilidade**.

## Como funciona

```text
Fotos rotuladas + recortes em water_regions.json
        ↓
Extração de histogramas RGB
        ↓
Treinamento e gravação de models/water_model.joblib
        ↓
Foto enviada → seleção da água → mesmas características → Limpo ou Sujo
```

1. `water_regions.json` relaciona cada foto de `ImagemCopos/` ao rótulo correto e às coordenadas do recorte da água. Atualmente há **9 fotos: 4 limpas e 5 sujas**.
2. O treino conta quantos pixels têm cada intensidade de vermelho, verde e azul no recorte. Cada histograma é dividido pelo número de pixels, para que o tamanho do recorte não determine o resultado.
3. O modelo calcula a diferença entre as intensidades médias dos canais **vermelho e verde**. O `ColorIntervalClassifier` aprende uma faixa para as fotos limpas, incluindo água de cor neutra. Valores fora da faixa são classificados como `sujo`.
4. A aplicação Flask carrega o modelo salvo ao iniciar. Ela recebe a foto, aplica a orientação EXIF, recorta a área selecionada, extrai as mesmas características e mostra o resultado com uma prévia do recorte.

O modelo padrão **não é SVM, Random Forest ou KNN**. Esses algoritmos aparecem em `water_classifier/model.py` e `evaluate.py` por causa de um experimento anterior com `res.csv`. O modelo atual é o classificador de intervalo de cor treinado com as fotos.

## Como instalar e executar

Execute os comandos abaixo **na pasta do projeto**. O projeto foi testado com Python 3.14; use uma versão compatível com as dependências de `requirements.txt` e com `venv` disponível.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Avalie as fotos rotuladas e treine o modelo atual:

```bash
python evaluate_images.py
python train.py
```

O treinamento grava `models/water_model.joblib`. A avaliação grava `reports/image_evaluation.json`. Para executar os testes e iniciar a interface:

```bash
python -m pytest -q
python app.py
```

Abra **http://127.0.0.1:5000** no navegador. Envie uma imagem JPEG, PNG ou WebP de até 8 MB e arraste sobre a foto para marcar somente a água. Evite borda do copo, fundo, mesa e reflexos. O recorte precisa ter pelo menos 32 × 32 pixels. Se treinar novamente enquanto o servidor estiver aberto, reinicie `python app.py` para carregar o novo arquivo do modelo.

Se o ambiente virtual já existir, basta ativá-lo e executar os comandos de avaliação, treino ou aplicação. Também é possível chamar `.venv/bin/python train.py` diretamente, sem ativação.

## Como adicionar fotos de treinamento

1. Coloque a foto em `ImagemCopos/`.
2. Adicione uma entrada a `water_regions.json` com `path`, `label` (`limpo` ou `sujo`) e `crop`. As quatro coordenadas do recorte são frações de 0 a 1, na ordem **esquerda, topo, direita, base**, medidas após aplicar a orientação EXIF.
3. Execute `python evaluate_images.py` e `python train.py` novamente; depois reinicie a aplicação.

Evite repetir a mesma foto com nomes diferentes: isso torna a avaliação artificialmente otimista. Inclua fotos de copos, câmeras, fundos e iluminações variados.

## Avaliação e limites

`evaluate_images.py` deixa **uma foto inteira fora do treinamento** a cada rodada e compara a previsão com o rótulo. O relatório atual registra 9 acertos em 9 fotos. Esse número **não mede o desempenho em fotos novas**: a característica, os recortes e um ajuste do modelo foram escolhidos após examinar estas mesmas imagens. Para medir generalização, é preciso coletar mais fotos e reservar um conjunto externo antes de ajustar o modelo.

## Arquivos principais

| Arquivo | Função |
| --- | --- |
| `water_regions.json` e `ImagemCopos/` | Fotos, rótulos e recortes de treino |
| `water_classifier/features.py` | Recorte e extração das características RGB |
| `water_classifier/image_training.py` | Classificador atual e leitura das fotos |
| `water_classifier/model.py` | Treino e gravação do modelo; algoritmos antigos do CSV |
| `train.py` | Comando de treinamento |
| `evaluate_images.py` | Avaliação por foto |
| `app.py`, `templates/`, `static/` | Interface web e classificação |
| `models/water_model.joblib` | Modelo carregado por padrão pela aplicação |

O experimento anterior com `res.csv` está descrito em [reports/KDD.md](reports/KDD.md). Seus resultados não representam o modelo atual.
