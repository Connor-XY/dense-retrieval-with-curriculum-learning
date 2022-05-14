from models import NewGuesserWithCT

guesser = NewGuesserWithCT()
guesser.load(model_path='new_models/new_guesser/checkpoint-79000')
guesser.test()
