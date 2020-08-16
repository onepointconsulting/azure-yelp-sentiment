import logging
import azure.functions as func

import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Prepare the transformers pipeline
tokenizer = AutoTokenizer.from_pretrained("gilf/english-yelp-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("gilf/english-yelp-sentiment")
nlp_sentiment = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    sentence = req.params.get('sentence')
    if not sentence:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            sentence = req_body.get('sentence')

    res = nlp_sentiment(sentence)[0]
    res['stars'] = convert_label(res)
    if sentence:
        return func.HttpResponse(json.dumps({"sentence": sentence, "analysis": res}), mimetype="application/json")
    else:
        return func.HttpResponse(
             "Please pass a sentence on the query string or in the request body!",
             status_code=400
        )

star_map = {
    "LABEL_0": 1,
    "LABEL_1": 2,
    "LABEL_2": 3,
    "LABEL_3": 4,
    "LABEL_4": 5
}

def convert_label(sentiment_result):
    '''
    Converts the sentiment result expressed as labels to the number of stars.
    '''
    return star_map[sentiment_result['label']]