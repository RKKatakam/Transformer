from model import *

encoder = Encoder()
assert encoder.decode(encoder.encode("Hello, world!")) == "Hello, world!"

print("All tests passed!")