"""Módulo de detecção de emoções usando Watson NLP ou um analisador local de fallback."""

import os
from typing import Any, Dict

try:
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_watson.natural_language_understanding_v1 import (
        EmotionOptions,
        Features,
    )
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False

EMOTION_KEYWORDS = {
    "happy": ["happy", "joy", "delight", "excited", "glad", "good", "love", "pleased", "smile"],
    "sadness": ["sad", "unhappy", "down", "blue", "upset", "miserable", "gloom", "cry", "sorrow"],
    "anger": ["angry", "mad", "furious", "irritated", "annoyed", "hate", "rage", "frustrated"],
    "fear": ["afraid", "fear", "scared", "anxious", "worried", "nervous", "panic", "terrified"],
    "disgust": ["disgusted", "gross", "nauseated", "repulsed", "sick", "horrible", "dirty"],
}


def _simple_emotion_analysis(text: str) -> Dict[str, float]:
    lower_text = text.lower()
    counts = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            counts[emotion] += lower_text.count(keyword)

    total_hits = sum(counts.values())
    base_probability = 0.1
    if total_hits == 0:
        return {emotion: base_probability for emotion in counts}

    weighted = {
        emotion: base_probability + count / total_hits * 0.9
        for emotion, count in counts.items()
    }
    return weighted


def emotion_detector(text: str) -> Dict[str, Any]:
    """Detecta emoções no texto de entrada e retorna um resultado formatado."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    text = text.strip()
    if WATSON_AVAILABLE:
        api_key = os.environ.get("WATSON_API_KEY", "")
        service_url = os.environ.get("WATSON_URL", "")
        if api_key and service_url:
            authenticator = IAMAuthenticator(api_key)
            nlu = NaturalLanguageUnderstandingV1(
                version="2021-08-01",
                authenticator=authenticator,
            )
            nlu.set_service_url(service_url)
            response = nlu.analyze(
                text=text,
                features=Features(emotion=EmotionOptions()),
            ).get_result()
            emotion_scores = response["emotion"]["document"]["emotion"]
        else:
            emotion_scores = _simple_emotion_analysis(text)
    else:
        emotion_scores = _simple_emotion_analysis(text)

    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    confidence = float(emotion_scores[primary_emotion])
    return {
        "input_text": text,
        "emotions": emotion_scores,
        "primary_emotion": primary_emotion,
        "confidence": confidence,
    }
