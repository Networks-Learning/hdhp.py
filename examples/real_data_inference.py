import matplotlib
import hdhp
import json
import timeit
from datetime import datetime
import random
import operator
import codecs
import re

matplotlib.use('Agg')


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

    new_data = {}

    with open(old_file_path) as old_file:

        old_data = json.load(old_file)

        for identifier in old_data:
            paper = old_data.get(identifier)
            document = db[metadata_collection].find_one({'identifier': {'$in': [identifier]}})
            new_abstract = document["description"][0]

            paper_abstract = new_abstract.lower()
            paper_abstract = re.sub("{|}|=|;|,|\?|\d+|'", "", paper_abstract)
            paper_abstract = re.sub("\(|\)|:|\d+|\.", " ", paper_abstract)
            paper_abstract = ' '.join(
                [word.strip() for word in paper_abstract.split() if word not in stopwords and len(word) > 1])
            paper["abstract"] = paper_abstract

            paper_title = document["title"].lower()
            paper_title = re.sub("{|}", "", paper_title)
            paper_title = re.sub(":|;|,|\?|\.", " ", paper_title)
            paper_title = ' '.join([word.strip() for word in paper_title.split() if word not in stopwords])
            paper["title"] = paper_title

            citations = paper['citations']
            new_citations = []

            for index in range(len(citations)):

                citation = citations[index]
                author = citation['author']
                new_author = []

                for item in author:
                    item = item.strip().lower()
                    if len(item) < 2:
                        continue
                    if 'et al.' in item:
                        item = item[0: item.index('et al.')]
                    if item.startswith(','):
                        continue
                    if ':' in item:
                        item = item[0: item.index(':')]

                    if '{' in item and len(item.split(' ')) > 3:
                        item = item[0: item.index('{')]
                    if '$' in item:
                        item = item[0: item.index('$')]
                    if 'title' in item:
                        continue
                    if ',' in item:
                        splitted = item.split(',')
                        if len(splitted) > 1:
                            item = splitted[1].strip() + ' ' + splitted[0].strip()
                        else:
                            item = splitted[0].strip()

                    splitted_item = item.split(' ')

                    new_item = ''

                    for temp in splitted_item:
                        new_item += temp.strip() + '#'

                    new_item = new_item[0:-1]
                    new_author.append(new_item)

                citation['author'] = new_author

                new_citations.append(citation)

                paper['citations'] = new_citations
            new_data[identifier] = paper

    json_file = open(new_file_path, 'w')
    json.dump(new_data, json_file, indent=0)
    json_file.close()
    client.close()
    print("Execution Time: " + str(timeit.default_timer() - start))


def json_file_to_events(json_file_path, vocab_types):
    start = timeit.default_timer()

    events = list()
    json_data = json.load(open(json_file_path))
    times = {}

    for identifier in json_data:
        paper = json_data.get(identifier)
        times[identifier] = paper["time"]

    sorted_times = sorted(times.items(), key=operator.itemgetter(1))
    counter = 0
    unique_authors = {}

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

        for author in authors:
            if author not in unique_authors:
                unique_authors[author] = counter
                authors_ids.append(counter)
                counter += 1
            else:
                authors_ids.append(unique_authors[author])

        vocabularies = {"docs": paper[vocab_types[0]], "auths": authors_vocabs.strip()}

        paper["author_ids"] = authors_ids
        event = (paper["time"], vocabularies, paper["author_ids"], [])
        events.append(event)

    print("Number of events: " + str(len(events)))
    print("Execution Time: " + str(timeit.default_timer() - start))
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


def authors_info(dataset_file_path):
    with open(dataset_file_path) as input_file:

        json_data = json.load(input_file)
        unique_authors = {}
        papers_per_user = {}
        events_per_user = {}
        counter = 0

        for identifier in json_data:

            paper = json_data.get(identifier)
            authors = paper["author"]

            for author in authors:
                if author not in unique_authors:
                    unique_authors[author] = counter
                    counter += 1

                if unique_authors[author] not in papers_per_user:
                    papers_per_user[unique_authors[author]] = 1
                else:
                    papers_per_user[unique_authors[author]] += 1

            if unique_authors.get(authors[0]) not in events_per_user:
                events_per_user[unique_authors.get(authors[0])] = 1
            else:
                events_per_user[unique_authors.get(authors[0])] += 1

        sorted_num_events = sorted(events_per_user.items(), key=operator.itemgetter(1))

        with open("num_events_per_user.txt", 'w') as out_file:

            for item in sorted_num_events:
                out_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')

        sorted_num_papers = sorted(papers_per_user.items(), key=operator.itemgetter(1))
        with open("num_papers_per_author.txt", 'w') as out_file:

            for item in sorted_num_papers:
                out_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')

        print("Number of unique authors: " + str(len(unique_authors)))


def infer(raw_events, indices, num_particles, alpha_0, mu_0, omega, use_cousers=False):
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
    print("Convert to process - time: " + str(timeit.default_timer() - start))

    return inf_process


def main():
    real_data_file_path = "../Real_Dataset/new_CS_arXiv_real_data.json"

    # priors to control the time dynamics of the events
    alpha_0 = (4.0, 0.5)  # prior for excitation
    mu_0 = (8, 0.25)  # prior for base intensity
    omega = 5  # decay kernel
    num_particles = 10

    vocab_types = ["abstract", "auths"]
    events = json_file_to_events(real_data_file_path, vocab_types)
    number_of_events = len(events)

    print("Number of events: " + str(number_of_events))

    cases = {1: ([0], False),
             2: ([0, 1], False),
             3: ([0, 1], True)}

    for case in [3, 2, 1]:
        print "Case: {0}".format(case)
        indices, use_cousers = cases[case]

        print("Start inferring.....")
        inferred_process = infer(events[: number_of_events], indices, num_particles, alpha_0, mu_0, omega,
                                 use_cousers=use_cousers)
        print("End inferring...")

        with open("real_data_results/" + "Case{0}".format(case) + "/title_base_rates_" + str(
                number_of_events) + "_" + vocab_types[0] + ".tsv", "w") as output_file:
            for key in inferred_process.mu_per_user:
                output_file.write("\t".join([str(key), str(inferred_process.mu_per_user[key])]) + "\n")

        with open("real_data_results/" + "Case{0}".format(case) + "/title_est_time_kernels_" + str(
                number_of_events) + "_" + vocab_types[0] + ".tsv", "w") as output_file:
            for key in inferred_process.time_kernels:
                output_file.write("\t".join([str(key), str(inferred_process.time_kernels[key])]) + "\n")

        clusters = inferred_process.show_annotated_events()
        with codecs.open("real_data_results/" + "Case{0}".format(case) + "/title_annotated_events_" + str(
                number_of_events) + "_" + vocab_types[0] + ".txt", "w", encoding="utf-8") as output_file:
            output_file.write(clusters)

        dist = inferred_process.show_pattern_content()
        with codecs.open("real_data_results/" + "Case{0}".format(case) + "/title_pattern_content_" + str(
                number_of_events) + "_" + vocab_types[0] + ".txt", "w", encoding="utf-8") as output_file:
            output_file.write(dist)

        predicted_labels = [e[1] for e in inferred_process.annotatedEventsIter()]

        with open("real_data_results/" + "Case{0}".format(case) + "/title_patterns_" + str(number_of_events) + "_" +
                          vocab_types[0] + ".tsv",
                  "w") as output_file:
            for i in xrange(len(predicted_labels)):
                output_file.write(str(predicted_labels[i]) + "\n")


if __name__ == "__main__":
    main()
