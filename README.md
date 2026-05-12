# Detector de Fadiga em Tempo Real

Projeto de visão computacional para detectar sinais visuais de fadiga usando OpenCV, MediaPipe Face Mesh e regras temporais explicáveis.

Esta primeira versão implementa um MVP local: webcam ou vídeo, landmarks faciais, cálculo de EAR/MAR, PERCLOS em janela móvel, score de fadiga, alerta visual/sonoro e log CSV por execução.

## O problema

Detectar fadiga em vídeo não deve depender de um único frame. Uma piscada normal pode fechar os olhos por poucos milissegundos e não representa sonolência. Por isso, este projeto analisa sinais persistentes em uma janela temporal.

Sinais usados no MVP:

- Fechamento dos olhos por EAR.
- PERCLOS, porcentagem de frames com olhos fechados na janela.
- Bocejo aproximado por MAR.
- Piscadas longas.
- Inclinação da cabeça por roll estimado.
- Ausência ou instabilidade de detecção facial.

## Estados do sistema

- `Atento`: sinais dentro do padrão da janela.
- `Atencao`: score intermediário, possível início de fadiga.
- `Fadiga`: score alto persistente.
- `Rosto ausente`: poucos frames válidos na janela.

## Estrutura

```text
detector_fadiga/
  app.py
  batch_process.py
  dashboard.py
  evaluate_labels.py
  requirements.txt
  configs/
    default.yaml
  src/
    alert.py
    capture.py
    features.py
    fatigue_rules.py
    landmarks.py
    evaluation.py
    pipeline.py
    plots.py
    reporting.py
    utils.py
    visualization.py
  tests/
    test_batch_process.py
    test_evaluation.py
    test_features.py
    test_fatigue_rules.py
    test_plots.py
    test_reporting.py
    test_utils.py
  outputs/
    batches/
    charts/
    evaluations/
    logs/
    reports/
    videos/
  data/
    labels/
    samples/
```

## Como rodar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Rode com webcam:

```bash
python app.py --source webcam
```

Rode com um vídeo:

```bash
python app.py --source caminho/para/video.mp4
```

Rode sem janela gráfica, útil para processar vídeo e gerar log:

```bash
python app.py --source caminho/para/video.mp4 --headless
```

Rode vídeo offline salvando overlay e relatórios:

```bash
python app.py --source caminho/para/video.mp4 --headless --save-video --no-sound
```

Por padrão, arquivos de vídeo usam o tempo do próprio vídeo para calcular janelas temporais. Webcam usa tempo real.

Abra o dashboard de análise dos logs:

```bash
streamlit run dashboard.py
```

O dashboard também lê o exemplo `data/samples/sample_session.csv`.

## Deploy no Streamlit Cloud

Use as seguintes configurações ao publicar:

- Repository: `LucasSGarrido/detector_fadiga`
- Branch: `main`
- Main file path: `dashboard.py`
- Python version: `3.12`

O projeto usa MediaPipe e OpenCV. No Streamlit Community Cloud, escolha a versão do Python em
**Advanced settings** antes de criar o app. Evite Python 3.14 para este projeto, porque algumas
dependências de visão computacional ainda podem não ter distribuições compatíveis.

Salve o vídeo processado com overlay:

```bash
python app.py --source caminho/para/video.mp4 --save-video
```

Processe uma pasta de vídeos:

```bash
python batch_process.py --input-dir data/raw --save-video --continue-on-error
```

Avalie um log com labels manuais:

```bash
python evaluate_labels.py --log outputs/logs/session.csv --labels data/labels/meu_video_labels.csv
```

Limite a execução para teste rápido:

```bash
python app.py --source webcam --max-frames 300
```

## Configuração

Os thresholds ficam em `configs/default.yaml`.

Principais parâmetros:

- `ear_closed`: valor abaixo do qual o olho é considerado fechado.
- `mar_yawn`: valor acima do qual a boca pode indicar bocejo.
- `perclos_attention`: início de atenção.
- `perclos_fatigue`: nível forte de fadiga.
- `window.seconds`: tamanho da janela temporal.
- `alert.min_duration_seconds`: tempo mínimo de score alto antes de alertar.
- `alert.cooldown_seconds`: intervalo mínimo entre alertas sonoros.
- `video.preserve_aspect_ratio`: mantém a proporção de vídeos verticais/horizontais ao redimensionar.

## Logs

Cada execução gera um CSV em `outputs/logs/`.

Quando `--no-report` não for usado, cada execução também gera:

- JSON em `outputs/reports/`.
- Markdown em `outputs/reports/`.
- Gráficos HTML em `outputs/charts/`.

O processamento em lote gera um CSV de resumo em `outputs/batches/`.

Avaliações com labels geram JSON e Markdown em `outputs/evaluations/`.

Campos principais:

- `frame_id`
- `face_detected`
- `ear_mean`
- `mar`
- `head_roll_deg`
- `perclos`
- `blink_count`
- `long_blink_count`
- `yawn_count`
- `fatigue_score`
- `state`
- `alert_triggered`
- `fps`
- `latency_ms`
- `reasons`

## Dashboard

O arquivo `dashboard.py` lê CSVs de `outputs/logs/`, CSVs de `data/samples/` ou um CSV enviado manualmente.

Ele mostra:

- Modo `Analisar vídeo` com upload direto na sidebar.
- Modo `Webcam ao vivo` usando webcam local via OpenCV.
- Modo `Revisar sessão` para logs já gerados.
- Resumo da sessão.
- Filtros por estado e score.
- Distribuição de estados.
- Timeline de score e PERCLOS.
- Timeline de EAR e MAR.
- Tabela de alertas.
- Avaliação com labels, quando um CSV de labels é selecionado.
- Player do vídeo processado, quando existir.
- Dados brutos da execução.

## Pipeline reutilizável

O processamento principal fica em `src/pipeline.py`.

Ele é usado por:

- `app.py`
- `dashboard.py`
- `batch_process.py`

Isso evita duplicação entre CLI, upload de vídeo, webcam e processamento em lote.

## Labels manuais

Para avaliar um vídeo, crie um CSV com este formato:

```csv
start_seconds,end_seconds,label
0.0,10.0,Atento
10.0,15.0,Fadiga
15.0,22.0,Atencao
```

Labels aceitos no projeto:

- `Atento`
- `Atencao`
- `Fadiga`
- `Rosto ausente`

Existe um exemplo em `data/labels/sample_labels.csv`.

## Fluxo sem webcam

Como a webcam não é obrigatória, o fluxo principal pode ser:

1. Colocar vídeos em `data/raw/`.
2. Rodar `python app.py --source caminho/para/video.mp4 --headless --save-video --no-sound`.
3. Abrir `streamlit run dashboard.py`.
4. Revisar `outputs/logs/`, `outputs/reports/`, `outputs/charts/` e `outputs/videos/`.
5. Ajustar thresholds em `configs/default.yaml`.
6. Reprocessar o vídeo.
7. Criar labels manuais e rodar `evaluate_labels.py`.

Para vários vídeos, usar `batch_process.py`.

## Testes

Rode:

```bash
pytest
```

Os testes atuais cobrem:

- EAR.
- MAR.
- Configuração padrão.
- Regras temporais para `Atento`, `Fadiga` e `Rosto ausente`.
- Resumo de logs e filtro de alertas.
- Comandos de lote para vídeos offline.
- Avaliação com labels manuais.
- Geração de gráficos HTML.

## Limitações da versão atual

- Usa MediaPipe Face Mesh no MVP para evitar download manual de modelo externo.
- A interface Streamlit atual analisa logs, mas ainda não faz webcam em tempo real.
- Head pose ainda usa apenas roll simples, não pitch/yaw completo.
- MAR pode confundir fala ou riso com bocejo.
- Thresholds ainda precisam de calibração por pessoa e por ambiente.
- Ainda não há avaliação com dataset público.

## Próximos passos

1. Testar com vídeo real e ajustar thresholds.
2. Gravar GIF curto do funcionamento usando vídeo processado.
3. Testar dashboard com logs reais.
4. Adicionar modo de análise offline com timeline mais detalhada.
5. Avaliar com vídeos anotados.
6. Comparar regras com baseline de machine learning.
7. Testar webcam quando houver câmera disponível.
