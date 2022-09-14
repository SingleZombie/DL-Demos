import random

from babel.dates import format_date
from faker import Faker

faker = Faker()
format_list = [
    'short', 'medium', 'long', 'full', 'd MMM YYY', 'd MMMM YYY', 'dd/MM/YYY',
    'dd-MM-YYY', 'EE d, MMM YYY', 'EEEE d, MMMM YYY'
]

if __name__ == '__main__':
    for format in format_list:
        date_obj = faker.date_object()
        print(f'{format}:', date_obj,
              format_date(date_obj, format=format, locale='en'))


def generate_date():
    format = random.choice(format_list)
    date_obj = faker.date_object()
    formated_date = format_date(date_obj, format=format, locale='en')
    return formated_date, date_obj


def generate_date_data(count, filename):
    with open(filename, 'w') as fp:
        for _ in range(count):
            formated_date, date_obj = generate_date()
            fp.write(f'{formated_date}\t{date_obj}\n')


def load_date_data(filename):
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        return [line.strip('\n').split('\t') for line in lines]


# generate_date_data(50000, 'dldemos/attention/train.txt')
# generate_date_data(10000, 'dldemos/attention/test.txt')
