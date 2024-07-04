import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

device = torch.device('cuda', 0)


def pipline_demo():
    classifier = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english', device=device)
    result = classifier(
        [
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )

    print(result)


def what_happened():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    outputs = model(**inputs)
    print(outputs.logits.shape)
    print(outputs.logits)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
    print(model.config.id2label)


def main() -> None:
    what_happened()


if __name__ == '__main__':
    main()
