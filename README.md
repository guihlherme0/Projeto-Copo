# Projeto Copo

Aplicação web em Python/Flask que recebe uma foto de um copo e classifica a **aparência da água** como `Limpo` ou `Sujo`. O resultado é visual e experimental: uma foto não informa se a água é potável nem detecta contaminantes invisíveis.

O projeto mantém dois modelos para finalidades diferentes:

| Modo | Dados usados no treino | Região da foto analisada | Modelo salvo |
| --- | --- | --- | --- |
| **Padrão da interface** | 9 fotos rotuladas de `ImagemCopos/` | Metade central da área marcada pelo usuário | `models/water_image_interval.joblib` |
| **Experimento com `res.csv`** | Todas as 50 linhas do CSV do professor | Foto inteira | `models/water_model.joblib` |

## Instalação

Na raiz do projeto, com Python 3.14:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Os dois arquivos de modelo já estão no projeto. Para treinar novamente, siga os comandos do modo desejado abaixo. **Reinicie o servidor após treinar**, pois ele carrega o modelo ao iniciar.

## Executar o modo padrão: modelo das fotos

```bash
python app.py
```

Abra <http://127.0.0.1:5000>. Envie uma imagem JPEG, PNG ou WebP de até 8 MB e marque aproximadamente a região da água. A resposta mostra `Limpo` ou `Sujo` e o recorte efetivamente analisado.

Para reproduzir a avaliação e o treinamento desse modelo:

```bash
python evaluate_images.py
python train.py
python app.py
```

`water_regions.json` contém o rótulo e a região de água de cada uma das nove fotos: quatro `limpo` e cinco `sujo`. No treino, cada foto fornece dois exemplos de histograma RGB: a região anotada e sua metade central. Na interface, usa-se a metade central da seleção feita pelo usuário para reduzir a influência de fundo, borda do copo e reflexos. Isso corrigiu o caso de `sujo12.jpg`, que era previsto como `Limpo` quando a seleção abrangia toda a foto e agora retorna `Sujo`.

O pipeline normaliza os três histogramas RGB e calcula a diferença entre as médias de vermelho e verde. O classificador aprende um intervalo dessa diferença para a classe `limpo`; valores fora dele recebem `sujo`. A avaliação deixa **uma foto inteira de fora por vez**, inclusive as duas versões dessa foto, e registrou 9/9 acertos nas fotos disponíveis. Esse resultado é **exploratório**: a característica e os recortes foram escolhidos após inspecionar essas fotos. É preciso testar fotos independentes para medir o desempenho em novas cenas.

## Executar o modo exigido com `res.csv`

O CSV fornecido pelo professor contém **50 registros**, com 14 `limpo` e 36 `sujo`. Cada linha tem 768 contagens: `r0`–`r255`, `g0`–`g255` e `b0`–`b255`; a coluna `class` é o rótulo. O servidor extrai essas 768 contagens da **foto inteira** quando usa o modelo do CSV. O hash SHA-256 do arquivo original é `2639238a463b61909fb3115a0026517a3a3f2e825503feaf0d33a8d33935ae10`.

Para reproduzir todas as etapas e iniciar a interface com o vencedor:

```bash
python evaluate.py
python train.py --source csv
WATER_MODEL_PATH=models/water_model.joblib python app.py
```

`evaluate.py` compara **SVM RBF, Random Forest e KNN 3** com as mesmas cinco dobras estratificadas repetidas cinco vezes (`random_state=42`). O pipeline de cada candidato é ajustado somente no treino de cada dobra. O critério de escolha é o maior **F1 macro médio por dobra**; em empate, maior recall de `limpo`.

| Algoritmo | F1 macro por dobra ± desvio padrão | Acurácia agregada | Recall de `limpo` |
| --- | ---: | ---: | ---: |
| **SVM RBF — vencedor** | **0,748 ± 0,114** | 0,784 | **0,786** |
| Random Forest | 0,738 ± 0,121 | 0,788 | 0,729 |
| KNN 3 | 0,689 ± 0,146 | 0,748 | 0,629 |

A acurácia agrega 250 previsões fora da dobra: os mesmos 50 registros aparecem uma vez em cada repetição. Os resultados completos, inclusive matrizes de confusão, ficam em `reports/csv_model_selection.json`. `train.py --source csv` confere o hash do CSV nesse relatório e depois ajusta o **SVM vencedor em todas as 50 linhas, sem divisão final**. As métricas acima vêm da validação cruzada, não do modelo final treinado em todos os dados.

Para **testar o KNN na interface** sem substituir o SVM vencedor, treine-o também nas 50 linhas e salve em outro arquivo:

```bash
python train.py --source csv --algorithm knn_3 --model models/water_knn.joblib
WATER_MODEL_PATH=models/water_knn.joblib python app.py
```

Esse comando não altera o relatório de seleção nem o modelo padrão. O KNN analisa a foto inteira e, nas nove fotos disponíveis, também marcou as quatro fotos limpas como `Sujo`.

### Etapas do KDD

1. **Seleção:** usar `res.csv` e identificar as 768 características RGB e a classe.
2. **Pré-processamento:** validar cabeçalho, rótulos, contagens e total de pixels por canal em `water_classifier/data.py`. Não há valores ausentes nem linhas duplicadas no arquivo atual.
3. **Transformação:** dividir cada histograma pelo total de pixels do seu canal. SVM e KNN usam também `StandardScaler`, ajustado dentro de cada dobra.
4. **Mineração:** comparar os três algoritmos com `evaluate.py`.
5. **Avaliação:** escolher pelo F1 macro médio, examinar recall e matrizes de confusão.
6. **Uso:** treinar o vencedor no CSV inteiro com `python train.py --source csv` e executá-lo pela variável `WATER_MODEL_PATH`.

Os totais por canal no CSV são 12.000.000 ou 12.979.200 pixels e aparecem em proporções diferentes nas duas classes. A normalização remove o número bruto de pixels, mas padrões de captura ainda podem estar associados ao rótulo. Como não há identificador de foto, copo ou sessão, a validação não pôde separar imagens por grupo de origem.

## Como a classificação funciona e seus limites

`app.py` recebe a foto, corrige sua orientação EXIF e usa `water_classifier/features.py` para contar os 256 níveis de cada canal RGB. O modelo salvo indica se espera o recorte da água ou a foto inteira. O servidor aplica o mesmo pipeline de transformação usado no treino e mostra o rótulo e a região analisada. Os arquivos da interface são `templates/index.html`, `static/style.css` e `static/app.js`.

As somas dos histogramas no CSV são compatíveis com fotos completas, mas **não foram fornecidas as 50 imagens originais nem o programa que gerou o CSV**. Os notebooks `AguaLimpaCode.ipynb` e `AguaLimpaCode_feature_selection.ipynb`, fornecidos pelo professor, mostram o treino com as colunas já prontas; não mostram a extração RGB das fotos. O primeiro testa árvore de decisão, MLP, SVM linear e KNN sobre contagens brutas. O segundo testa também seleção de 200 atributos por `chi2` e 500 por ANOVA, mas sua saída salva tem **61 linhas**, outra versão da base; suas métricas não podem ser atribuídas ao CSV atual de 50 linhas. `evaluate_notebook_methods.py` reproduz configurações representativas no CSV atual, sem alterar o vencedor da avaliação principal.

Nas nove fotos disponíveis, os três algoritmos treinados no CSV classificaram todas como `Sujo`: acertaram as cinco sujas e erraram as quatro limpas. Recortar a água, redimensionar ou permutar os canais RGB não corrigiu esse erro do SVM. O modelo padrão das fotos melhora esses exemplos, mas **não substitui o experimento exigido com `res.csv`**. Também faltam identificadores de copo ou sessão e fotos independentes para avaliar generalização. O modelo deve ser usado apenas como demonstração de classificação visual.

## Comandos adicionais e arquivos principais

```bash
python -m pytest -q                 # testes
python audit_feature_contract.py    # diagnóstico do CSV contra as fotos
python evaluate_notebook_methods.py # reprodução de métodos dos notebooks no CSV atual
```

Os diagnósticos opcionais gravam JSON em `reports/`. `water_classifier/model.py` define e salva os pipelines; `water_classifier/image_training.py` carrega as fotos rotuladas e treina o classificador visual; `evaluate_images.py` avalia esse classificador deixando uma foto de fora por vez. Carregue arquivos `.joblib` apenas de origem confiável.
