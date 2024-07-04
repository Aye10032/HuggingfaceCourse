from transformers import pipeline
import torch

device = torch.device('cuda', 0)


def test_pipline():
    classifier = pipeline("sentiment-analysis", device=device)
    result = classifier(
        ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
    )

    print(result)


def zero_shot_classification():
    classifier = pipeline("zero-shot-classification", device=device)
    result = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )

    print(result)


def text_generation():
    generator = pipeline("text-generation", model="distilgpt2", device=device)
    result = generator(
        "In this course, we will teach you how to",
        truncation=True,
        num_return_sequences=2,
        max_length=500
    )

    print(result)


def mask_filling():
    unmasker = pipeline("fill-mask", device=device)
    result = unmasker("This course will teach you all about <mask> models.", top_k=2)

    print(result)


def named_entity_recognition():
    ner = pipeline("ner", grouped_entities=True)
    result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

    print(result)


def qa():
    question_answerer = pipeline("question-answering", device=device)
    result = question_answerer(
        question="Where do I work?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )

    print(result)


def summary():
    summarizer = pipeline("summarization", device=device)
    result = summarizer(
        """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """,
        max_length=100,
        min_length=50
    )

    print(result)


def translate():
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en", device=device)
    result = translator("Ce cours est produit par Hugging Face.")

    print(result)


def main() -> None:
    translate()


if __name__ == '__main__':
    main()
