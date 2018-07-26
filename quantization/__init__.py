import os, sys

path = os.path.split(os.path.realpath(__file__))[0]
path = path + "/.."

sys.path.append(path)
