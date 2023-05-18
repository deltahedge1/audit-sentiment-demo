from transformers import pipeline

class EmotionClassifier:
    def __init__(self, model="j-hartmann/emotion-english-distilroberta-base"):
        self.model = model
        self.classifier = pipeline("text-classification", model=model, top_k=None)

    def inference(self, text):
        emotions = self.classifier(text)
        emotions = self._flatten_dict(emotions[0])
        positive = emotions["joy"] + emotions["surprise"]
        negative = emotions["anger"] + emotions["sadness"] + emotions["fear"] + emotions["disgust"]
        emotions = {**emotions, **emotions}
        emotions["sentiment_positive"] = positive
        emotions["sentiment_negative"] = negative
        emotions["sentiment_neutral"] = emotions["neutral"]
        return emotions

    def _flatten_dict(self, data):
        return {item['label']: item['score'] for item in data}