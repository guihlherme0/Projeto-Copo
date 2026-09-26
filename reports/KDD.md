# Relatório KDD — classificação visual da água

> Registro histórico do experimento com `res.csv`. O modelo padrão atual usa as fotos em `ImagemCopos/`; veja `README.md` e `reports/image_evaluation.json` para o treino atual. As métricas abaixo não descrevem o modelo atual.

## 1. Entendimento e seleção

O repositório estava vazio. Não havia material da aula disponível no projeto ou arquivos de aula identificáveis junto ao CSV fornecido. A fonte é `res.csv` (SHA-256 `2639238a463b61909fb3115a0026517a3a3f2e825503feaf0d33a8d33935ae10`). Há **50 linhas, 769 colunas**. As entradas são `r0`–`r255`, `g0`–`g255`, `b0`–`b255`; `class` é o alvo, com **14 `limpo` e 36 `sujo`**. Cada linha contém três histogramas de 256 contagens inteiras. Em cada linha, os três canais somam o mesmo número de pixels, sempre **12.000.000 (27 linhas)** ou **12.979.200 (23 linhas)**. Isso sustenta a interpretação de um histograma RGB por imagem, provavelmente da imagem inteira, mas o arquivo não contém a foto nem explica a extração, a ordem efetiva dos canais ou o critério de rotulagem.

## 2. Pré-processamento e qualidade

Não há campos ausentes, linhas duplicadas, contagens negativas ou não inteiras. Todas as 50 linhas têm as 769 colunas esperadas e os canais de cada uma têm totais iguais. Não foi necessário excluir linhas nem imputar valores. A seleção preservou os 768 atributos, na ordem exata do cabeçalho. Dentro de cada `Pipeline`, cada canal é dividido por sua própria soma de pixels; os valores tornam-se frequências entre 0 e 1. Os modelos sensíveis à escala também recebem `StandardScaler`, ajustado apenas nos registros de treino de cada dobra.

O tamanho da imagem é um possível atalho para o rótulo: entre as linhas `limpo`, 12 de 14 têm 12.979.200 pixels; entre as `sujo`, 25 de 36 têm 12.000.000. A normalização remove o total bruto, mas padrões de captura, iluminação, fundo e resolução ainda podem estar correlacionados à classe. Como não há identificador de foto, copo ou sessão, não foi possível verificar ou impor separação por grupo. Também não há fotos para identificar regiões da água ou detectar imagens quase repetidas visualmente.

## 3. Mineração e avaliação

Foram comparados cinco algoritmos com parâmetros fixados antes da avaliação. Método: **RepeatedStratifiedKFold, 5 dobras × 5 repetições, `random_state=42`**, total de 25 ajustes por algoritmo. Cada validação tem 10 registros. O `Pipeline` completo é clonado e ajustado de novo em cada dobra. `limpo` é a classe minoritária. O critério de escolha é maior **F1 macro médio por dobra**; em eventual empate, maior recall de `limpo`. Não houve busca de hiperparâmetros. O uso das mesmas dobras para escolher o algoritmo implica possível otimismo; uma estimativa independente exigiria novas fotos ou validação externa agrupada.

Ambiente usado: Python 3.14, NumPy 2.5.3, SciPy 1.18.1 e scikit-learn 1.9.1; versões fixadas em `requirements.txt`.

Parâmetros: maioria (`DummyClassifier`, `most_frequent`); regressão logística (`C=0.1`, `class_weight=balanced`, `max_iter=3000`, escala padrão); SVM (`kernel=rbf`, `C=1`, `class_weight=balanced`, escala padrão); floresta (`n_estimators=200`, `min_samples_leaf=2`, `class_weight=balanced`, `random_state=42`); kNN (`n_neighbors=3`, escala padrão). A normalização RGB ocorre em todos os pipelines.

Os valores de acurácia, precisão, recall e F1 por classe a seguir são agregados das **250 previsões fora da dobra** de cada algoritmo. Cada uma das 50 linhas é prevista 5 vezes, uma em cada repetição. A coluna de variação é a média ± desvio padrão do F1 macro calculado separadamente nas 25 dobras. Assim, as previsões agregadas não são 250 fotos independentes.

| Algoritmo | Acurácia | Limpo P / R / F1 | Sujo P / R / F1 | F1 macro agregado | F1 macro por dobra ± DP |
|---|---:|---:|---:|---:|---:|
| Maioria | 0,720 | 0,000 / 0,000 / 0,000 | 0,720 / 1,000 / 0,837 | 0,419 | 0,418 ± 0,013 |
| Regressão logística | 0,788 | 0,605 / 0,700 / 0,649 | 0,876 / 0,822 / 0,848 | 0,749 | 0,735 ± 0,128 |
| **SVM RBF** | **0,784** | **0,585 / 0,786 / 0,671** | **0,904 / 0,783 / 0,839** | **0,755** | **0,748 ± 0,114** |
| Floresta aleatória | 0,788 | 0,600 / 0,729 / 0,658 | 0,885 / 0,811 / 0,846 | 0,752 | 0,738 ± 0,121 |
| kNN 3 | 0,748 | 0,543 / 0,629 / 0,583 | 0,846 / 0,794 / 0,819 | 0,701 | 0,689 ± 0,146 |

Matrizes de confusão agregadas, **linhas reais e colunas previstas na ordem [`limpo`, `sujo`]**:

| Algoritmo | Matriz |
|---|---|
| Maioria | `[[0, 70], [0, 180]]` |
| Regressão logística | `[[49, 21], [32, 148]]` |
| SVM RBF | `[[55, 15], [39, 141]]` |
| Floresta aleatória | `[[51, 19], [34, 146]]` |
| kNN 3 | `[[44, 26], [37, 143]]` |

O **SVM RBF** venceu pelo F1 macro médio por dobra (0,748), com o maior recall da classe minoritária (0,786). Sua acurácia ficou ligeiramente abaixo da regressão e da floresta. A diferença de F1 para os outros dois é pequena em relação à variação entre dobras; este resultado indica preferência neste experimento, não superioridade estatística comprovada. O modelo não produz probabilidade calibrada; a aplicação não exibe confiança.

### Verificação com fotos novas

Depois da implementação, a foto `limpo2.jpg` enviada pelo usuário, visualmente de água limpa, foi classificada como `sujo` pelo SVM original. `sujo2.jpg`, uma segunda foto disponível, também recebeu `sujo`. Em recortes centrais da água, as decisões numéricas do SVM ficaram praticamente iguais (`0,77556` para ambas), embora as imagens sejam diferentes. No espaço de entrada do SVM, após normalização e `StandardScaler`, a distância da foto limpa ao exemplo de treino mais próximo foi **1497,46**, e a da segunda foto **136,12**; o percentil 95 das distâncias de cada registro ao vizinho mais próximo no treino é **33,29**. O classificador, portanto, estava extrapolando para fotografias fora do padrão observado. Um ensaio com histogramas de 8, 16, 32 e 64 faixas por canal ainda classificou ambas como `sujo`; esses ensaios não foram incorporados à avaliação comparativa acima nem ao modelo final.

Um Random Forest treinado nas 50 linhas foi testado pela mesma rota Flask e salvo separadamente em `models/water_random_forest.joblib`. Ele classificou `limpo2.jpg` e `sujo2.jpg` como `Sujo` tanto na foto inteira quanto nos recortes centrais da água. Na validação do CSV, teve acurácia agregada de 0,788 e F1 macro médio por dobra de 0,738 ± 0,121. Esse teste mostra que trocar apenas o algoritmo por Random Forest não corrigiu o erro observado na foto limpa.

A interface foi mantida com somente duas saídas, `Limpo` e `Sujo`, conforme o requisito do usuário. Ela sempre devolve uma classe para uma imagem válida. O erro observado em `limpo2.jpg` continua sendo uma limitação conhecida: a classificação binária obrigatória não aumenta a capacidade de generalização do modelo. As duas fotos novas não foram incluídas no treinamento.

## 4. Interpretação, treinamento final e limites

Após a escolha, o pipeline SVM foi ajustado **uma única vez com as 50 linhas de `res.csv`**, sem divisão. O arquivo `models/water_model.joblib` guarda o pipeline, o nome do algoritmo, a ordem das características, as classes, o total de registros e o hash da fonte. **Todas as métricas acima vêm da validação cruzada, não do modelo final ajustado em todos os dados.**

Na aplicação, o usuário marca a região da água e o servidor calcula contagens RGB inteiras nos mesmos 768 índices. A normalização é a mesma do treino. Entretanto, o CSV não registra se os histogramas originais correspondiam a um recorte de água, à foto inteira ou a outro processamento. A seleção reduz influência do fundo na foto nova, mas pode mudar a distribuição em relação ao treino. Sem fotos originais e protocolo de captura, não é possível medir essa diferença nem garantir generalização para fotos de celular. A água também pode estar contaminada sem alteração visual. Uma classe exibida deve ser lida apenas como **classe visual experimental**.

Para uma avaliação de uso real, obter fotos originais e IDs de copo/foto/sessão, confirmar o mapeamento RGB e a região representada em cada linha, padronizar captura e rotulagem, e coletar um conjunto externo capturado pela interface. Então refazer a validação separando os grupos de origem e medir o desempenho nessa distribuição.
