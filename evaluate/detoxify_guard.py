from detoxify import Detoxify

# each model takes in either a string or a list of strings

model = Detoxify('original', device='cuda')

print('example text', model.predict('example text'))

print('a woman', model.predict('a woman'))
print('a naked woman', model.predict('a naked woman'))

print('a butcher', model.predict('a butcher'))
print('a bloody butcher', model.predict('a bloody butcher'))