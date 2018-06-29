import re
from nltk.stem import PorterStemmer


def process_citation(citation):
    author = citation['author']
    new_author = []

    for item in author:
        item = re.sub("{\\\\ A}", "", item).strip()

        if item.endswith(", A"):
            item = item[0:-3]
        if item.endswith(" A"):
            item = item[0:-2]

        item = item.strip().lower()
        if len(item) < 2:
            continue

        if 'et al.' in item:
            item = item[0: item.index('et al.')].strip()
        if item.startswith(','):
            continue
        if ':' in item:
            item = item[0: item.index(':')].strip()
        if 'physic' in item or 'ieee' in item or 'nature' in item:
            continue
        if ', {a}' in item:
            item = item[0: item.index(', {a}')].strip()
        if '{a}' in item:
            item = item[0: item.index('{a}')].strip()
        if '(' in item:
            item = item[0: item.index('(')]

        item = re.sub('{\\\ n}', 'n', item)
        item = re.sub('{\\\ a}', 'a', item)
        item = re.sub('\\\,', ' ', item)
        item = re.sub('\\\ ', '', item)
        item = re.sub('\\\\', '', item)

        if '$' in item:
            item = item[0: item.index('$')].strip()
        if 'title' in item:
            continue

        item = re.sub("{ | }", "", item)
        item = re.sub("{|}", "", item)

        item = item.strip()

        if ',' in item:
            splitted = item.split(',')
            if len(splitted) > 1:
                item = splitted[1].strip() + ' ' + splitted[0].strip()
            else:
                item = splitted[0].strip()

        splitted_item = item.split(' ')

        if not splitted_item[0].strip().endswith('.'):
            if len(splitted_item) == 2 or len(splitted_item) == 3:
                for j in range(len(splitted_item) - 1):
                    if '-' in splitted_item[j]:
                        temp = splitted_item[j].split('-')
                        if temp[0] != '':
                            splitted_item[j] = temp[0][0].strip() + '.-'
                        if temp[1] != '':
                            splitted_item[j] += temp[1][0].strip() + '.'
                    else:
                        if splitted_item[j].strip() != '':
                            splitted_item[j] = splitted_item[j].strip()[0] + '.'

        new_item = ''

        for temp in splitted_item:
            if temp.strip() != "":
                new_item += temp.strip() + '#'

        new_item = new_item[0:-1]
        if new_item.endswith('.'):
            new_item = new_item[0:-1]

        if len(new_item) > 4:
            new_author.append(new_item)

    citation['author'] = new_author
    return citation


def process_abstract_title(abstract, title, stopwords):
    ps = PorterStemmer()

    paper_abstract = abstract.lower()
    paper_abstract = re.sub("=|;|,|\?|\d+|'|\+|\^|\[|\]", "", paper_abstract)
    paper_abstract = re.sub("{|}|\\\\\S*|\$\S*\$|\(|\)|:|\d+|\.", " ", paper_abstract)

    all_words = paper_abstract.split()
    paper_abstract = ""

    for word in all_words:
        word = word.strip()
        if len(word) > 1 and word not in stopwords:
            paper_abstract += ps.stem(word) + ' '

    paper_title = title.lower()
    paper_title = re.sub("\d+|\(|\)|:|;|,|\?|\.|'|\$\S*\$|\\\\\S*|{|}", " ", paper_title)
    paper_title = re.sub("\+|=|\^|\[|\]", "", paper_title)

    all_words = paper_title.split()
    paper_title = ""

    for word in all_words:
        word = word.strip()
        if len(word) > 1 and word not in stopwords:
            paper_title += ps.stem(word) + ' '

    return paper_abstract, paper_title


def process_authors(authors, unique_authors, counter):
    authors_ids = []

    for author in authors:

        author = author.lower().strip()
        if ',' in author:
            split_author = author.split(',')
            author = split_author[1].strip() + ' ' + split_author[0].strip()
        split_author = author.split(' ')

        if not split_author[0].endswith('.') and len(split_author) > 1:

            for i in range(len(split_author) - 1):
                if '-' in split_author[i]:
                    temp = split_author[i].split('-')

                    if temp[0] != '':
                        split_author[i] = temp[0][0].strip() + '.-'
                    if temp[1] != '':
                        split_author[i] += temp[1][0].strip() + '.'
        author = ""
        for temp in split_author:
            author += temp + ' '
        author = author.strip()

        if author not in unique_authors:
            unique_authors[author] = counter
            authors_ids.append(counter)
            counter += 1
        else:
            authors_ids.append(unique_authors[author])

    return authors_ids, counter
