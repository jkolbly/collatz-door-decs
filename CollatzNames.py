import math

def load_names(filename):
  with open(filename) as f:
    return [name.strip() for name in f.readlines()]
  
def get_name_objects(filename):
  return [CollatzName(name) for name in load_names(filename)]
  
def gen_sequence(start):
  cur = start
  seq = []
  while cur != 1:
    seq.append(cur)
    if cur % 2 == 0:
      cur //= 2
    else:
      cur = 3 * cur + 1
  seq.append(1)
  return seq

def char_to_int(char):
  c = ord(char)
  if c >= 65 and c <= 90:
    return c - 64
  elif c >= 97 and c <= 122:
    return c - 96
  print(f'Cannot convert {char} to int.')

def name_to_int(name: str):
  return math.prod([char_to_int(char) for char in name])

class CollatzName:
  def __init__(self, name: str):
    self.name = name
    self.char_seq = [char_to_int(char) for char in name]
    self.int = name_to_int(name)
    self.collatz_seq = gen_sequence(self.int)

  def __str__(self):
    return f"<CollatzNames.CollatzName object: {self.name} at {hex(id(self))}>"
    
  def __repr__(self):
    return str(self)

if __name__ == "__main__":
  names = load_names("names.csv")
  for name in names:
    print(f'{name}:')
    print([char_to_int(c) for c in name])
    print(gen_sequence(name_to_int(name)))
    print()