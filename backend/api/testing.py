import os
import pathlib

print(os.path.dirname(os.path.abspath(__file__)))
print(pathlib.Path(__file__).parent.resolve())

a = pathlib.Path(__file__).parent.resolve()
print(a/"assests"/"proccessed.csv")