from models import NewGuesser

guesser = NewGuesser()
guesser.load(model_path='models/new_guesser')
guesser.test()