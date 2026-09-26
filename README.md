# Projeto Copo

Para uma explicação da ideia, do modelo e dos comandos, consulte [GUIA_DO_PROJETO.md](GUIA_DO_PROJETO.md).

Aplicação Flask para classificar **aparência visual** de água limpa ou suja em uma foto de copo. O resultado não mede potabilidade nem detecta contaminantes invisíveis.

## Instalação e execução

Testado com Python 3.14. No diretório do projeto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate_images.py
python train.py
python -m pytest -q
python app.py
```

Abra `http://127.0.0.1:5000`. Envie JPEG, PNG ou WebP de até 8 MB e marque uma região **dentro da água**, evitando borda do copo, mesa, fundo, espuma e reflexos. A seleção precisa ter pelo menos 32 × 32 pixels. O servidor aplica a orientação EXIF antes de recortar a foto.

## Modelo atual

O modelo padrão em `models/water_model.joblib` usa as nove fotos de `ImagemCopos/`: quatro rotuladas `limpo` e cinco `sujo`. As regiões analisadas estão documentadas em `water_regions.json`. O treino conta as intensidades RGB dos pixels da região, normaliza os histogramas pelo tamanho da região e calcula a diferença entre as médias dos canais vermelho e verde. O classificador aprende uma faixa para as fotos limpas, incluindo água de cor neutra, e classifica as demais cores como sujas. O arquivo do modelo guarda o hash do manifesto e das fotos para reproduzir o treino.

`reports/image_evaluation.json` registra uma validação deixando uma foto inteira fora do treino a cada rodada: **9 acertos em 9 fotos**. Essa taxa é preliminar: a característica, os recortes e o ajuste para água de cor neutra foram escolhidos depois de olhar essas mesmas fotos. Ela **não é uma estimativa independente** de desempenho em fotos novas. A iluminação, o balanço de branco, o fundo, o recipiente e o ponto do recorte podem mudar a diferença de cor; o modelo pode errar mesmo quando devolve `Limpo` ou `Sujo`. Para medir generalização, colete mais fotos de copos e sessões diferentes e reserve algumas desde o início para teste externo.

## Análise anterior do CSV

O modelo anterior foi treinado apenas com os 50 histogramas de `res.csv`. Nas oito fotos de exemplo, ele devolvia `sujo` para todas, inclusive as três fotos limpas. Os histogramas do CSV não registram as fotos de origem nem a região usada, e suas cores diferem das fotos atuais. O experimento anterior permanece em `evaluate.py`, `reports/evaluation.json` e `reports/KDD.md`; suas métricas não devem ser usadas para descrever o modelo atual.

Para reproduzir o experimento legado em um arquivo separado:

```bash
python evaluate.py --csv res.csv --output reports/evaluation.json
python train.py --csv res.csv --algorithm svm_rbf --model models/water_csv_legacy.joblib
```

Defina `WATER_MODEL_PATH` para usar outro arquivo de modelo na aplicação. Carregue arquivos `.joblib` apenas de origem confiável.
