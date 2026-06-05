# Emotion Detector

Projeto final: Detector de Emoções usando Flask e Watson NLP.

- Repositório: https://github.com/cesarfavero/ibm-python-flask.git
- Aplicação: `EmotionDetection`
- Servidor Flask: `server.py`
- Testes unitários: `test_emotion_detection.py`

## Como rodar

1. Crie um ambiente virtual Python.
2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute os testes:
   ```bash
   python -m unittest test_emotion_detection.py
   ```
4. Inicie o servidor:
   ```bash
   python server.py
   ```

## Sobre

A aplicação analisa texto de entrada e retorna uma saída estruturada sobre emoções detectadas. Quando disponível, ela utiliza a biblioteca `ibm_watson` para análise com Watson NLP; caso contrário, usa um analisador local de fallback.
