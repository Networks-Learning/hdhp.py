import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import hdhp
import json
import timeit
from datetime import datetime
import random
import operator
import codecs
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import datetime
import os
from dateutil import relativedelta


def plot_papers_per_year(dataset_file_path):
    """
    This function plot number of papers published in each year
    :param dataset_file_path: Dataset File Path
    :return: Nothing.
    """

    start = timeit.default_timer()
    json_data = json.load(open(dataset_file_path))
    years = {}

    for identifier in json_data:
        paper = json_data[identifier]
        paper_year = paper['year']
        if paper_year in years:
            years[paper_year] += 1
        else:
            years[paper_year] = 1
    x = []
    y = []

    for key in years:
        x.append(int(key))
        y.append(years.get(key))

    width = 1 / 1.5
    fig, ax = plt.subplots()
    ax.bar(x, y, width, color="c", linewidth=0.4)
    ax.set_ylabel('Number of Papers')
    ax.set_title('Publish Year')
    fig = plt.gcf()
    fig.savefig('years.png')
    fig.clf()
    plt.close(fig)
    print("Plotting number of papers for each year in " + str(timeit.default_timer() - start) + " seconds")


def plot_pattern_popularity(file_path):
    with open(file_path) as input_file:
        patterns = {}
        lines = input_file.readlines()
        for line in lines:
            line = line.strip()
            if line not in patterns:
                patterns[line] = 1
            else:
                patterns[line] += 1

        sorted_patterns = sorted(patterns, key=lambda t: t[1] * 1)
        print(sorted_patterns)


def find_important_words(dataset_file_path, new_file_path):
    start = timeit.default_timer()

    json_data = json.load(open(dataset_file_path))
    docs = []
    new_data = {}

    for identifier in json_data:
        docs.append(json_data.get(identifier).get('abstract') + ' ' + json_data.get(identifier).get('title'))

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

    feature_names = tfidf_vectorizer.get_feature_names()
    dense = tfidf_matrix.todense()

    for index, (identifier, paper) in enumerate(json_data.items()):

        doc = dense[index].tolist()[0]
        word_scores = [pair for pair in zip(range(0, len(doc)), doc) if pair[1] > 0]

        sorted_non_zero_scores = sorted(word_scores, key=lambda t: t[1] * -1)
        sorted_features = []

        for item in sorted_non_zero_scores:
            sorted_features.append(feature_names[item[0]])

        paper['sorted_features'] = sorted_features
        new_data[identifier] = paper

    with open(new_file_path, 'w') as outout_file:
        json.dump(new_data, outout_file, indent=1)

    print ("tfidf process - execution time: " + str(timeit.default_timer() - start))


def create_mongodb_connection_info(db_server, db_name, db_user, db_password):
    """
    This function returns the mongoDB connection info.
    :param db_server: address of mongoDB server
    :param db_name: mongoDB database name
    :param db_user: mongoDB user
    :param db_password: user's password
    :return: mongoDB connection info.
    """
    db_connection = "mongodb://" + db_user + ":" + db_password + "@" + db_server + "/" + db_name
    return db_connection


def metadata_to_database(metadata_file_path, db_connection_info, metadata_collection):
    """
    This function gets the metadata of all papers as a xml file, and inserts them into a 'MongoDB' database.
    :param metadata_file_path: The address of the metadata file (xml file).
    :param db_connection_info: database connection information (server, username, password)
    :param metadata_collection: Name of the metadata collection
    :return: Nothing.
    """

    from pymongo import MongoClient
    import xml.etree.ElementTree as ElementTree

    client = MongoClient(db_connection_info)  # Connect to the MongoDB
    db = client.arXiv  # Gets the related database

    for event, elem in ElementTree.iterparse(metadata_file_path):

        if elem.tag == "metadata":
            mongoDB_document = {}
            for child in elem.iter():  # Build a document for each record's metadata
                current_tag = child.tag
                if current_tag == "metadata" or current_tag == "{http://www.openarchives.org/OAI/2.0/oai_dc/}dc":
                    continue
                current_tag = current_tag.replace("{http://purl.org/dc/elements/1.1/}", "").strip()
                child_text = child.text
                if child_text is None:
                    continue
                child_text = child_text.strip().replace("\n", " ")
                if current_tag == "type" or current_tag == "title" or current_tag == "language":
                    mongoDB_document[current_tag] = child_text
                else:
                    if current_tag in mongoDB_document:
                        mongoDB_document[current_tag] = mongoDB_document.get(current_tag) + [child_text]
                    else:
                        mongoDB_document[current_tag] = [child_text]

            db[metadata_collection].insert_one(mongoDB_document)  # Insert the document in to the database
            elem.clear()
    client.close()


def clean_real_data(db_connection_info, old_file_path, new_file_path, metadata_collection, stopwords_file_path):
    """
    This function clean the text of the abstract, title, and citations of each paper in the real dataset. Modified data
    stored in a new json file.
    :param db_connection_info: MongoDB connection info
    :param old_file_path: Real data file path (format: json)
    :param new_file_path: New data file path (format: json)
    :param metadata_collection: metadata collection
    :param stopwords_file_path: Stopwords file path
    :return: Nothing.
    """

    from pymongo import MongoClient

    start = timeit.default_timer()

    with open(stopwords_file_path) as stopwords_file:
        stopwords = stopwords_file.readlines()

    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].strip()

    client = MongoClient(db_connection_info)  # Connect to the MongoDB
    db = client.arXiv  # Gets the related database

    ps = PorterStemmer()

    new_data = {}
    unique_authors = {}
    counter = 0

    with open(old_file_path) as old_file:

        old_data = json.load(old_file)

        for identifier in old_data:
            paper = old_data.get(identifier)
            document = db[metadata_collection].find_one({'identifier': {'$in': [identifier]}})
            new_abstract = document["description"][0]

            paper_abstract = new_abstract.lower()
            paper_abstract = re.sub("=|;|,|\?|\d+|'|\+|\^|\[|\]", "", paper_abstract)
            paper_abstract = re.sub("{|}|\\\\\S*|\$\S*\$|\(|\)|:|\d+|\.", " ", paper_abstract)

            all_words = paper_abstract.split()
            paper_abstract = ""

            for word in all_words:
                word = word.strip()
                if len(word) > 1 and word not in stopwords:
                    paper_abstract += ps.stem(word) + ' '

            paper["abstract"] = paper_abstract

            paper_title = document["title"].lower()
            paper_title = re.sub("\d+|\(|\)|:|;|,|\?|\.|'|\$\S*\$|\\\\\S*|{|}", " ", paper_title)
            paper_title = re.sub("\+|=|\^|\[|\]", "", paper_title)

            all_words = paper_title.split()
            paper_title = ""

            for word in all_words:
                word = word.strip()
                if len(word) > 1 and word not in stopwords:
                    paper_title += ps.stem(word) + ' '

            paper["title"] = paper_title

            citations = paper['citations']
            new_citations = []

            for index in range(len(citations)):

                citation = citations[index]
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

                new_citations.append(citation)

                paper['citations'] = new_citations

                authors = paper['author']
                authors_ids = []

                for author in authors:

                    author = author.lower().strip()
                    if ',' in author:
                        splitted_author = author.split(',')
                        author = splitted_author[1].strip() + ' ' + splitted_author[0].strip()
                    splitted_author = author.split(' ')

                    if not splitted_author[0].endswith('.'):

                        for i in range(len(splitted_author) - 1):
                            if '-' in splitted_author[i]:
                                temp = splitted_author[i].split('-')

                                if temp[0] != '':
                                    splitted_author[i] = temp[0][0].strip() + '.-'
                                if temp[1] != '':
                                    splitted_author[i] += temp[1][0].strip() + '.'
                    author = ""
                    for temp in splitted_author:
                        author += temp + ' '
                    author = author.strip()

                    if author not in unique_authors:
                        unique_authors[author] = counter
                        authors_ids.append(counter)
                        counter += 1
                    else:
                        authors_ids.append(unique_authors[author])

                paper['authors_ids'] = authors_ids
            new_data[identifier] = paper

    with open(new_file_path, 'w') as output_file:
        json.dump(new_data, output_file, indent=0)

    client.close()
    print("Cleaning Data - Execution Time: " + str(timeit.default_timer() - start))


def json_file_to_events(json_file_path, vocab_types, num_words, base_year, selected_authors):
    from datetime import datetime
    start = timeit.default_timer()
    base_time = datetime.strptime(base_year + '-01-01', '%Y-%m-%d')

    events = list()
    json_data = json.load(open(json_file_path))
    times = {}

    for identifier in json_data:
        paper = json_data.get(identifier)
        if int(paper['year']) > int(base_year):
            paper_time = datetime.strptime(paper['date'][0], '%Y-%m-%d')
            diff = relativedelta.relativedelta(paper_time, base_time)
            num_months = diff.years * 12 + diff.months
            times[identifier] = num_months + random.uniform(0, 1)

    sorted_times = sorted(times.items(), key=operator.itemgetter(1))
    counter = 0
    unique_authors = {}

    unique_vocabs_1 = {}
    unique_vocabs_2 = {}

    for item in sorted_times:
        identifier = item[0]
        paper = json_data.get(identifier)

        authors_vocabs = ''

        for citation in paper["citations"]:
            authors = citation["author"]

            for author in authors:
                authors_vocabs += author.strip() + ' '

        authors = paper['author']
        authors_ids = []
        processed_authors = []

        for author in authors:

            author = author.lower().strip()

            if ',' in author:
                splitted_author = author.split(',')
                author = splitted_author[1].strip() + ' ' + splitted_author[0].strip()
            splitted_author = author.split(' ')

            if not splitted_author[0].endswith('.'):

                for i in range(len(splitted_author) - 1):
                    if '-' in splitted_author[i]:
                        temp = splitted_author[i].split('-')

                        if temp[0] != '':
                            splitted_author[i] = temp[0][0].strip() + '.-'
                        if temp[1] != '':
                            splitted_author[i] += temp[1][0].strip() + '.'
                    else:
                        splitted_author[i] = splitted_author[i].strip()[0] + '.'
            author = ""
            for temp in splitted_author:
                author += temp + ' '
            author = author.strip()
            processed_authors.append(author)
            if author not in unique_authors:
                unique_authors[author] = counter
                authors_ids.append(counter)
                counter += 1
            else:
                authors_ids.append(unique_authors[author])

        if authors_ids[0] not in selected_authors:
            continue

        if vocab_types[0] == "tfidf":
            n = min(num_words, len(paper["sorted_features"]))
            vocabularies = {"docs": ' '.join(paper["sorted_features"][0:n]), "auths": authors_vocabs.strip()}

            for vocab in paper["sorted_features"][0:n]:
                if vocab in unique_vocabs_1:
                    unique_vocabs_1[vocab] += 1
                else:
                    unique_vocabs_1[vocab] = 0

            for vocab in authors_vocabs.split(' '):
                if vocab in unique_vocabs_2:
                    unique_vocabs_2[vocab] += 1
                else:
                    unique_vocabs_2[vocab] = 0

        else:
            vocabularies = {"docs": paper[vocab_types[0]], "auths": authors_vocabs.strip()}
            for vocab in paper[vocab_types[0]].split(' '):
                if vocab in unique_vocabs_1:
                    unique_vocabs_1[vocab] += 1
                else:
                    unique_vocabs_1[vocab] = 0

            for vocab in authors_vocabs.split(' '):
                if vocab in unique_vocabs_2:
                    unique_vocabs_2[vocab] += 1
                else:
                    unique_vocabs_2[vocab] = 0

        paper["author_ids"] = authors_ids
        event = (times[identifier], vocabularies, paper["author_ids"], [])
        events.append(event)

    print("Number of events: " + str(len(events)))
    print("Number of unique authors: " + str(len(unique_authors)))
    print("Execution Time: " + str(timeit.default_timer() - start))
    print
    print("Docs vocab size: " + str(len(unique_vocabs_1)))
    print("Authors vocab size: " + str(len(unique_vocabs_2)))
    return events


def find_first_date(json_file_path):
    json_data = json.load(open(json_file_path))
    first_date = datetime.now()

    for identifier in json_data:
        paper = json_data.get(identifier)
        paper_time = datetime.strptime(paper["date"][0], '%Y-%m-%d')

        if paper_time < first_date:
            first_date = paper_time

    print(str(first_date))


def num_unique_authors(json_file_path):
    json_data = json.load(open(json_file_path))
    unique_authors = []

    for identifier in json_data:
        paper = json_data.get(identifier)
        for author in paper["author"]:
            unique_authors.append(author.strip())

    print("Number of all authors: " + str(len(unique_authors)))
    print("Number of unique authors: " + str(len(set(unique_authors))))


def get_number_of_authors(events):
    unique_authors = []

    for event in events:
        unique_authors += event[2]

    return len(set(unique_authors))


def maps_authors_to_ids(json_file_path):
    json_data = json.load(open(json_file_path))
    stopwords_file_path = "stopwords.txt"

    with open(stopwords_file_path) as stopwords_file:
        stopwords = stopwords_file.readlines()

    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].strip()

    new_file = "/NL/publications-corpus/work/new_CS_arXiv_real_data.json"

    base_time = datetime.strptime('1996-06-03', '%Y-%m-%d')

    counter = 0
    names_to_ids = {}
    new_json_data = {}

    for identifier in json_data:
        paper = json_data.get(identifier)

        authors = paper["author"]
        for author in authors:
            if author.strip() not in names_to_ids:
                counter += 1
                names_to_ids[author.strip()] = counter

    for identifier in json_data:
        paper = json_data.get(identifier)

        authors = paper["author"]
        ids = []
        for author in authors:
            ids.append(names_to_ids.get(author.strip()))
        paper_abstract = paper["abstract"].lower()
        paper_abstract = ' '.join([word for word in paper_abstract.split() if word not in stopwords])
        paper_abstract = re.sub(":|;|,|\?|\.", "", paper_abstract)
        paper["abstract"] = paper_abstract

        paper_title = paper["title"].lower()
        paper_title = ' '.join([word for word in paper_title.split() if word not in stopwords])
        paper_title = re.sub(":|;|,|\?|\.", "", paper_title)
        paper["title"] = paper_title

        paper['author_ids'] = ids

        paper_time = datetime.strptime(paper["date"][0], '%Y-%m-%d')
        time_diff = paper_time - base_time
        time = time_diff.total_seconds() / (3600 * 24) + random.uniform(0, 1)
        paper['time'] = time
        new_json_data[identifier] = paper

    with open(new_file, 'w') as output_file:
        json.dump(new_json_data, output_file, indent=1)
    print("Number of unique ids: " + str(len(names_to_ids)))
    return names_to_ids


def get_authors_with_n_papers(dataset_file_path, base_year, number_of_papers):
    json_data = json.load(open(dataset_file_path))
    unique_authors = {}
    papers_per_user = {}
    events_per_user = {}
    counter = 0

    for identifier in json_data:

        paper = json_data.get(identifier)
        if not paper['year'] > int(base_year):
            continue

        authors = paper['author']
        authors_ids = []

        for index, author in enumerate(authors):

            author = author.lower().strip()

            if ',' in author:
                splitted_author = author.split(',')
                author = splitted_author[1].strip() + ' ' + splitted_author[0].strip()
            splitted_author = author.split(' ')

            if not splitted_author[0].endswith('.'):

                for i in range(len(splitted_author) - 1):
                    if '-' in splitted_author[i]:
                        temp = splitted_author[i].split('-')

                        if temp[0] != '':
                            splitted_author[i] = temp[0][0].strip() + '.-'
                        if temp[1] != '':
                            splitted_author[i] += temp[1][0].strip() + '.'
                    else:
                        splitted_author[i] = splitted_author[i].strip()[0] + '.'
            author = ""
            for temp in splitted_author:
                author += temp + ' '
            author = author.strip()

            if author not in unique_authors:
                unique_authors[author] = counter
                authors_ids.append(counter)
                counter += 1
            else:
                authors_ids.append(unique_authors[author])

            if unique_authors[author] not in papers_per_user:
                papers_per_user[unique_authors[author]] = 1
            else:
                papers_per_user[unique_authors[author]] += 1

            if index == 0:
                if unique_authors.get(author) not in events_per_user:
                    events_per_user[unique_authors.get(author)] = 1
                else:
                    events_per_user[unique_authors.get(author)] += 1

    first_authors = [author for author in events_per_user if events_per_user[author] > number_of_papers]
    all_authors = [author for author in papers_per_user if papers_per_user[author] > number_of_papers]

    return first_authors, all_authors


def authors_info(dataset_file_path):
    with open(dataset_file_path) as input_file:

        json_data = json.load(input_file)
        unique_authors = {}
        papers_per_user = {}
        events_per_user = {}
        counter = 0

        for identifier in json_data:

            paper = json_data.get(identifier)
            authors = paper['author']
            authors_ids = []

            for index, author in enumerate(authors):

                author = author.lower().strip()

                if ',' in author:
                    splitted_author = author.split(',')
                    author = splitted_author[1].strip() + ' ' + splitted_author[0].strip()
                splitted_author = author.split(' ')

                if not splitted_author[0].endswith('.'):

                    for i in range(len(splitted_author) - 1):
                        if '-' in splitted_author[i]:
                            temp = splitted_author[i].split('-')

                            if temp[0] != '':
                                splitted_author[i] = temp[0][0].strip() + '.-'
                            if temp[1] != '':
                                splitted_author[i] += temp[1][0].strip() + '.'
                        else:
                            splitted_author[i] = splitted_author[i].strip()[0] + '.'
                author = ""
                for temp in splitted_author:
                    author += temp + ' '
                author = author.strip()

                if author not in unique_authors:
                    unique_authors[author] = counter
                    authors_ids.append(counter)
                    counter += 1
                else:
                    authors_ids.append(unique_authors[author])

                if unique_authors[author] not in papers_per_user:
                    papers_per_user[unique_authors[author]] = 1
                else:
                    papers_per_user[unique_authors[author]] += 1

                if index == 0:
                    if unique_authors.get(author) not in events_per_user:
                        events_per_user[unique_authors.get(author)] = 1
                    else:
                        events_per_user[unique_authors.get(author)] += 1

        sorted_num_events = sorted(events_per_user.items(), key=operator.itemgetter(1), reverse=True)

        with open("num_events_per_user.txt", 'w') as out_file:

            for item in sorted_num_events:
                out_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')

        sorted_num_papers = sorted(papers_per_user.items(), key=operator.itemgetter(1), reverse=True)
        with open("num_papers_per_author.txt", 'w') as out_file:

            for item in sorted_num_papers:
                out_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')

        print("Number of unique authors: " + str(len(unique_authors)))
        print("Events per user size: " + str(len(events_per_user)))
        print("Papers per user size: " + str(len(papers_per_user)))


def save_inferred_process(inf_process, vocab_types, directory_path):
    predicted_labels = [e[1] for e in inf_process.annotatedEventsIter()]
    popularity = {}

    for label in predicted_labels:
        if label not in popularity:
            popularity[label] = 0
        else:
            popularity[label] += 1

    for label in popularity:
        popularity[label] = popularity[label] * 1.0 / len(predicted_labels)

    with open(os.path.join(directory_path, vocab_types[0] + '_popularity_' + str(len(predicted_labels)) + '.json'),
              'w') as output_file:
        json.dump(popularity, output_file, indent=1)

    with open(os.path.join(directory_path, vocab_types[0] + '_time_kernels_' + str(len(predicted_labels)) + '.json'),
              'w') as output_file:
        json.dump(inf_process.time_kernels, output_file, indent=1)

    with open(os.path.join(directory_path, vocab_types[0] + '_table_history_' + str(len(predicted_labels)) + '.json'),
              'w') as output_file:
        json.dump(inf_process.table_history_per_user, output_file, indent=1)

    with open(os.path.join(directory_path, vocab_types[0] + '_time_history_' + str(len(predicted_labels)) + '.json'),
              'w') as output_file:
        json.dump(inf_process.time_history_per_user, output_file, indent=1)

    with open(os.path.join(directory_path, vocab_types[0] + '_dish_on_table_' + str(len(predicted_labels)) + '.json'),
              'w') as output_file:
        json.dump(inf_process.dish_on_table_per_user, output_file, indent=1)

    with open(os.path.join(directory_path, vocab_types[0] + '_mu_per_user_' + str(len(predicted_labels)) + '.json'),
              'w') as output_file:
        json.dump(inf_process.mu_per_user, output_file, indent=1)


def infer(raw_events, indices, num_particles, alpha_0, mu_0, omega, vocab_types, use_cousers=False):
    start = timeit.default_timer()

    types = ["docs", "auths"]

    # Inference
    types = [types[i] for i in indices]

    events = list()

    if use_cousers:
        for event in raw_events:
            events.append((event[0], {t: event[1][t] for t in types}, event[2], event[3]))
    else:
        for event in raw_events:
            events.append((event[0], {t: event[1][t] for t in types}, [event[2][0]], event[3]))

    particle, norms = hdhp.infer(events,
                                 alpha_0,
                                 mu_0,
                                 types,
                                 omega=omega,
                                 beta=1,
                                 threads=1,
                                 num_particles=num_particles,
                                 keep_alpha_history=True,
                                 seed=512)

    print("Execution time of calling infer function: " + str(timeit.default_timer() - start))
    start = timeit.default_timer()

    inf_process = particle.to_process()

    save_inferred_process(inf_process, vocab_types, "real_data_results/Inference/")

    print("Convert to process - time: " + str(timeit.default_timer() - start))

    return inf_process


def main():
    # plot_papers_per_year("../Real_Dataset/modified_2_CS_arXiv_real_data.json")
    # authors_info("../Real_Dataset/modified_2_CS_arXiv_real_data.json")
    real_data_file_path = "new_CS_arXiv_real_data.json"

    # priors to control the time dynamics of the events
    alpha_0 = (4.0, 0.5)  # prior for excitation
    mu_0 = (8, 0.25)  # prior for base intensity
    omega = 5  # decay kernel
    num_particles = 20

    db_connection_info = ""

    vocab_types = ["title", "auths"]
    print("Vocab Types: " + str(vocab_types))
    print("Number of Particles: " + str(num_particles))

    # clean_real_data(db_connection_info, real_data_file_path, "modified_CS_arXiv_real_data.json", "new_metadata", "stopwords.txt")
    # find_important_words( "../Real_Dataset/modified_CS_arXiv_real_data.json", "tfidf_CS_arXiv_real_data.json")

    dataset_file_path = "../Real_Dataset/modified_2_CS_arXiv_real_data.json"
    base_year = '2010'
    time_unit = "months"
    number_of_papers = 1

    first_authors, all_authors = get_authors_with_n_papers(dataset_file_path, base_year, number_of_papers)
    events = json_file_to_events(dataset_file_path, vocab_types, 10, base_year, first_authors)

    number_of_events = 100

    print("Number of events: " + str(number_of_events))

    cases = {1: ([0], False),
             2: ([0, 1], False),
             3: ([0, 1], True)}

    for case in [3]:
        print "Case: {0}".format(case)
        indices, use_cousers = cases[case]

        print("Start inferring.....")
        start = timeit.default_timer()
        inferred_process = infer(events[: number_of_events], indices, num_particles, alpha_0, mu_0, omega, vocab_types,
                                 use_cousers=use_cousers)
        print("End inferring in : " + str(timeit.default_timer() - start))

        with open("real_data_results/" + "Case{0}".format(case) + "/" + time_unit + "/" + vocab_types[
            0] + "_base_rates_" + str(
                number_of_events) + ".tsv", "w") as output_file:
            for key in inferred_process.mu_per_user:
                output_file.write("\t".join([str(key), str(inferred_process.mu_per_user[key])]) + "\n")

        with open("real_data_results/" + "Case{0}".format(case) + "/" + time_unit + "/" + vocab_types[
            0] + "_est_time_kernels_" + str(
                number_of_events) + ".tsv", "w") as output_file:
            for key in inferred_process.time_kernels:
                output_file.write("\t".join([str(key), str(inferred_process.time_kernels[key])]) + "\n")

        clusters = inferred_process.show_annotated_events()
        with codecs.open("real_data_results/" + "Case{0}".format(case) + "/" + time_unit + "/" + vocab_types[
            0] + "_annotated_events_" + str(
            number_of_events) + ".txt", "w", encoding="utf-8") as output_file:
            output_file.write(clusters)

        dist = inferred_process.show_pattern_content()
        with codecs.open("real_data_results/" + "Case{0}".format(case) + "/" + time_unit + "/" + vocab_types[
            0] + "_pattern_content_" + str(
            number_of_events) + ".txt", "w", encoding="utf-8") as output_file:
            output_file.write(dist)

        predicted_labels = [e[1] for e in inferred_process.annotatedEventsIter()]

        with open("real_data_results/" + "Case{0}".format(case) + "/" + time_unit + "/" + vocab_types[
            0] + "_patterns_" + str(
                number_of_events) + ".tsv",
                  "w") as output_file:
            for i in xrange(len(predicted_labels)):
                output_file.write(str(predicted_labels[i]) + "\n")


if __name__ == "__main__":
    main()
