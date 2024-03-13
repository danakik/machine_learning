import matplotlib.pyplot as plt
import string
from collections import Counter

def count_letter_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text = ''.join(filter(str.isalpha, text))
    text = text.lower()
    letter_count = Counter(text)
    sorted_letters = sorted(letter_count.items())

    letters, frequencies = zip(*sorted_letters)

    plt.bar(letters, frequencies)
    plt.title('Гістограма частоти появи літер')
    plt.xlabel('Літера')
    plt.ylabel('Частота')
    plt.show()

file_path = '1/text.txt'
count_letter_from_file(file_path)

