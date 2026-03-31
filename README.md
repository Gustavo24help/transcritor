# Transcrição de áudios WhatsApp Business

Script para:
1. Converter mídias exportadas (incluindo áudio de vídeo) com **ffmpeg**.
2. Transcrever com API OpenAI.
3. Inserir a transcrição em CSVs de conversa no ponto do `[áudio]`.

## Requisitos

- Python 3.10+
- `ffmpeg` no PATH (`ffmpeg -version`)
- Chave OpenAI com crédito

## Instalação

```bash
pip install openai tqdm
```

## Uso

```bash
python transcrever_audios.py \
  --media-dir "C:/Users/danie/Downloads/WhatsApp_Midias" \
  --csv-dir "C:/Users/danie/Downloads/WhatsApp_CSV" \
  --output-dir "C:/Users/danie/Downloads/WhatsApp_Transcrito" \
  --workers 5
```

Também pode usar a chave via variável de ambiente:

```bash
setx OPENAI_API_KEY "sk-..."
```

## Saídas

- `output/transcricoes.csv`: todas as transcrições
- `output/cache_transcricoes.json`: cache para retomada
- `output/*_com_transcricoes.csv`: conversas com campo `mensagem_final`

## Observações

- Se a coluna de mídia (`midia`, `media`, `arquivo` etc.) existir no CSV, o encaixe da transcrição é mais preciso.
- O script não altera os CSVs de origem; gera novos arquivos.
