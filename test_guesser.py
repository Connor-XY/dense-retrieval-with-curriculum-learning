from models import NewGuesser

guesser = NewGuesser()
guesser.load(model_path='models/new_guesser/checkpoint-64000')
guesser.test()