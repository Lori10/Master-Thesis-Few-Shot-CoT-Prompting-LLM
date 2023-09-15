from transformers import pipeline

generator = pipeline('text-generation', model='tensorcat/falcon-7b-instruct-8bit')
print(generator("Generate a story about a spaceship traveling through space.", max_length=200))

