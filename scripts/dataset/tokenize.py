import spacy as sp

nlp = sp.load("en_core_web_trf")

tokens = tokenizer("Hello my name is barry.")
print(tokens)
