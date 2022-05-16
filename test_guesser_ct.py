from models import NewGuesserWithCT

guesser = NewGuesserWithCT()
guesser.load(model_path='new_models/new_guesser')
guesser.test()
