from app import make_images
from random import randint
import json
import os

def main():
    for i in range(100):
        s1, s2, d1, d2 = make_images()
        s1 = json.dumps(s1)
        s2 = json.dumps(s2)

        i = randint(1, 1e9)
        while os.path.exists('static_%i_1.png' % i):
            i = randint(1, 1e9)

        with open('static/%i_1.png' % i, 'wb') as out:
            out.write(d1.decode('base64'))
        with open('static/%i_2.png' % i, 'wb') as out:
            out.write(d2.decode('base64'))
        with open('static/%i_1.json' % i, 'wb') as out:
            out.write(s1)
        with open('static/%i_2.json' % i, 'wb') as out:
            out.write(s2)

if __name__ == "__main__":
    main()
