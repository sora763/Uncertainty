# coding=utf-8
import sys
import os
sys.path.append("../")
from tools import getFilelist

path = sys.argv[1]
old_word = sys.argv[2]
new_word = sys.argv[3]
files = getFilelist(path, ".txt")
for f in files:
    read_f = open(f, "r")
    write_f = open("temp_file.txt", "w")
    for line in read_f.readlines():
        new_l = line.replace(old_word, new_word)
        write_f.write(new_l)

    read_f.close()
    write_f.close()
    os.remove(f)
    os.rename("temp_file.txt", f.split()[-1])
