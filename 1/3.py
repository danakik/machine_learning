import matplotlib.pyplot as plt


def plot_sentence_types_frequency(text):
    question = text.count('?')
    exclamation = text.count('!')
    ellipsis = text.count('...')
    period = text.count('.') - ellipsis * 3

    sentence_types = ['Знак питання', 'Знак оклику', 'Крапка', 'Трикрапка']
    frequencies = [question, exclamation, period, ellipsis]

    plt.bar(sentence_types, frequencies)
    plt.title('Гістограма частоти знаків у тексті')
    plt.xlabel('Тип знаку')
    plt.ylabel('Частота')
    plt.show()


text = """
    Це. І це? А це! Ні це не це... О! А це це.
"""

plot_sentence_types_frequency(text)
