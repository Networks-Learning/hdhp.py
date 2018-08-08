import matplotlib

matplotlib.use('Agg')

import timeit
import json
import hdhp
import datetime
import utility
import random
import operator
import codecs
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from dateutil import relativedelta
import dill
import pickle


def get_train_test_data(dataset_file_path, base_year, test_data_file_path, train_data_file_path):
    dataset = json.load(open(dataset_file_path))
    test_dataset = {}
    train_dataset = {}

    for identifier in dataset:
        paper = dataset.get(identifier)
        if int(paper['year']) >= int(base_year):
            test_dataset[identifier] = paper
        else:
            train_dataset[identifier] = paper

    with open(test_data_file_path, "w") as output_file:
        print("Test Dataset Size: " + str(len(test_dataset)))
        json.dump(test_dataset, output_file, indent=1)

    with open(train_data_file_path, "w") as output_file:
        print("Train Dataset Size: " + str(len(train_dataset)))
        json.dump(train_dataset, output_file, indent=1)


def get_year_based_events(dataset_file_path, base_year, new_dataset_file_path):
    from datetime import datetime

    base_time = datetime.strptime(base_year + '-01-01', '%Y-%m-%d')
    data = json.load(open(dataset_file_path))
    new_data = {}

    for identifier in data:
        paper = data.get(identifier)
        if int(paper['year']) > int(base_year):
            paper_time = datetime.strptime(paper['date'][0], '%Y-%m-%d')
            diff = relativedelta.relativedelta(paper_time, base_time)
            num_months = diff.years * 12 + diff.months
            paper['num_months'] = num_months
            paper['time_months'] = num_months + random.uniform(0, 1)
            paper['time_half_year'] = (num_months / 6.0) + random.uniform(0, 1)
            new_data[identifier] = paper

    with open(new_dataset_file_path, 'w') as output_file:
        print("Dataset Size: " + str(len(new_data)) + "\n")
        json.dump(new_data, output_file, indent=1)


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

    new_data = {}
    unique_authors = {}
    counter = 0

    with open(old_file_path) as old_file:

        old_data = json.load(old_file)

        for identifier in old_data:
            paper = old_data.get(identifier)
            new_paper = {}
            document = db[metadata_collection].find_one({'identifier': {'$in': [identifier]}})
            abstract = document["description"][0]
            title = document["title"].lower()
            new_paper['abstract'] = abstract
            new_paper['title'] = title

            paper_abstract, paper_title = utility.process_abstract_title(abstract, title, stopwords)
            new_paper["processed_abstract"] = paper_abstract
            new_paper["processed_title"] = paper_title

            citations = paper['citations']
            new_citations = []

            for index in range(len(citations)):
                processed_citation = utility.process_citation(citations[index])
                new_citations.append(processed_citation)

            new_paper['citations'] = new_citations

            authors = paper['author']
            authors_ids, counter = utility.process_authors(authors, unique_authors, counter)

            new_paper['authors_ids'] = authors_ids
            new_paper['author'] = paper.get('author')
            new_paper['file_name'] = paper.get('file_name')
            new_paper['tar_file'] = paper.get('tar_file')
            new_paper['year'] = paper.get('year')
            new_paper['date'] = paper.get('date')
            new_paper['subject'] = paper.get('subject')
            new_paper['time'] = paper.get('time')
            new_data[identifier] = new_paper

    with open(new_file_path, 'w') as output_file:
        print("Dataset Size: " + str(len(new_data)))
        json.dump(new_data, output_file, indent=0)

    with open("Statistics/authors_names_mapping.json", "w") as output_file:
        print("Number of Unique Authors: " + str(len(unique_authors)))
        json.dump(unique_authors, output_file, indent=1)

    client.close()
    print("Cleaning Data - Execution Time: " + str(timeit.default_timer() - start))


def json_file_to_events(json_file_path, vocab_types, num_words, selected_authors, author_index):
    start = timeit.default_timer()

    events = list()
    json_data = json.load(open(json_file_path))
    times = {}

    for identifier in json_data:
        paper = json_data.get(identifier)
        times[identifier] = paper['time_half_year']

    sorted_times = sorted(times.items(), key=operator.itemgetter(1))

    unique_vocabs_1 = {}
    unique_vocabs_2 = {}

    for item in sorted_times:
        identifier = item[0]
        paper = json_data.get(identifier)
        authors_ids = paper['authors_ids']

        if authors_ids[0] not in selected_authors and authors_ids[-1] not in selected_authors:
            continue

        authors_vocabs = ''

        for citation in paper["citations"]:
            authors = citation["author"]

            for author in authors:
                authors_vocabs += author.strip() + ' '

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
            vocabularies = {"docs": paper['processed_' + vocab_types[0]], "auths": authors_vocabs.strip()}
            for vocab in paper['processed_' + vocab_types[0]].split(' '):
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


def get_authors_with_n_papers(dataset_file_path, base_year, number_of_papers):
    json_data = json.load(open(dataset_file_path))
    papers_per_user = {}
    events_per_first_user = {}
    events_per_last_user = {}

    for identifier in json_data:

        paper = json_data.get(identifier)
        if not paper['year'] > int(base_year):
            continue

        authors_ids = paper.get('authors_ids')

        if authors_ids[0] not in events_per_first_user:
            events_per_first_user[authors_ids[0]] = 1
        else:
            events_per_first_user[authors_ids[0]] += 1

        if authors_ids[-1] not in events_per_last_user:
            events_per_last_user[authors_ids[-1]] = 1
        else:
            events_per_last_user[authors_ids[-1]] += 1

        for author_id in authors_ids:
            if author_id not in papers_per_user:
                papers_per_user[author_id] = 1
            else:
                papers_per_user[author_id] += 1

    with open("Statistics/num_events_per_first_user_" + str(number_of_papers) + '.txt', 'w') as output_file:
        sorted_authors = sorted(events_per_first_user.items(), key=lambda kv: kv[1], reverse=True)
        for item in sorted_authors:
            output_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')
        print ("Number of unique first authors: " + str(len(events_per_first_user)))

    with open("Statistics/num_papers_per_author_" + str(number_of_papers) + '.txt', 'w') as output_file:
        sorted_authors = sorted(papers_per_user.items(), key=lambda kv: kv[1], reverse=True)
        for item in sorted_authors:
            output_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')
        print ("Number of unique authors: " + str(len(papers_per_user)))

    with open("Statistics/num_events_per_last_user_" + str(number_of_papers) + '.txt', 'w') as output_file:
        sorted_authors = sorted(events_per_last_user.items(), key=lambda kv: kv[1], reverse=True)
        for item in sorted_authors:
            output_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')
        print ("Number of unique last authors: " + str(len(events_per_last_user)) + '\n')

    first_authors = [author for author in events_per_first_user if events_per_first_user[author] > number_of_papers]
    print("First authors with more than one paper: " + str(len(first_authors)))
    last_authors = [author for author in events_per_last_user if events_per_last_user[author] > number_of_papers]
    print("Last authors with more than one paper: " + str(len(last_authors)))
    all_authors = [author for author in papers_per_user if papers_per_user[author] > number_of_papers]
    print("Authors with more than one paper: " + str(len(all_authors)) + '\n')

    return first_authors, last_authors, all_authors


def infer(raw_events, indices, num_particles, alpha_0, mu_0, omega, author_index, use_cousers=False):
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
            events.append((event[0], {t: event[1][t] for t in types}, [event[2][author_index]], event[3]))

    particle, norms = hdhp.infer(events,
                                 alpha_0,
                                 mu_0,
                                 types,
                                 omega=omega,
                                 beta=1,
                                 threads=1,
                                 num_particles=num_particles,
                                 keep_alpha_history=True,
                                 seed=512,
                                 author_index=author_index)

    print("Execution time of calling infer function: " + str(timeit.default_timer() - start))
    start = timeit.default_timer()

    inf_process = particle.to_process()

    print("Convert to process - time: " + str(timeit.default_timer() - start))

    return inf_process


def main():
    raw_real_data_file_path = "Real_Dataset/new_CS_arXiv_real_data.json"

    # priors to control the time dynamics of the events
    alpha_0 = (4.0, 0.5)  # prior for excitation
    mu_0 = (8, 0.25)  # prior for base intensity
    omega = 5  # decay kernel
    num_particles = 20

    db_connection_info = ""

    vocab_types = ["title", "auths"]

    dataset_file_path = "Real_Dataset/final_CS_arXiv_real_data.json"
    # clean_real_data(db_connection_info, raw_real_data_file_path, dataset_file_path, "new_metadata", "stopwords.txt")

    base_year = '2010'
    time_unit = "months"
    number_of_papers = 1
    author_index = -1  # 0 for first authors and -1 for last authors

    final_dataset_file_path = "../Real_Dataset/data_after_year_" + str(base_year) + ".json"
    # get_year_based_events(dataset_file_path, base_year, final_dataset_file_path)

    test_data_base_year = '2016'
    train_data_file_path = "Real_Dataset/train_data_before_" + test_data_base_year + ".json"
    test_data_file_path = "Real_Dataset/test_data_after_" + test_data_base_year + ".json"

    # get_train_test_data(final_dataset_file_path, test_data_base_year, test_data_file_path, train_data_file_path)

    first_authors, last_authors, all_authors = get_authors_with_n_papers(final_dataset_file_path, base_year,
                                                                         number_of_papers)
    events = json_file_to_events(final_dataset_file_path, vocab_types, 10, all_authors, author_index)

    number_of_events = len(events)

    print("Number of events: " + str(number_of_events))
    print("Vocab Types: " + str(vocab_types))
    print("Number of Particles: " + str(num_particles) + '\n')

    cases = {1: ([0], False),
             2: ([0, 1], False),
             3: ([0, 1], True)}

    for case in [3]:
        print "Case: {0}".format(case)
        indices, use_cousers = cases[case]

        print("Start inferring.....")
        start = timeit.default_timer()
        inferred_process = infer(events[: number_of_events], indices, num_particles, alpha_0, mu_0, omega,
                                 author_index, use_cousers=use_cousers)

        with open("real_data_results/" + "Case{0}".format(case) + "/Inference/" +
                          vocab_types[0] + "_inferred_process_" + str(number_of_events) + ".pkl", "wb") as output_file:
            pickle.dump(inferred_process, output_file, pickle.HIGHEST_PROTOCOL)

        print("End inferring in : " + str(timeit.default_timer() - start) + '\n')
        print("***********************************************************************************\n")

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
            0] + "_patterns_" + str(number_of_events) + ".tsv", "w") as output_file:
            for i in xrange(len(predicted_labels)):
                output_file.write(str(predicted_labels[i]) + "\n")

        with open("real_data_results/" + "Case{0}".format(case) + "/" + time_unit + "/" + vocab_types[
            0] + "_events_" + str(number_of_events) + ".tsv", "w") as output_file:
            for event in inferred_process.annotatedEventsIter():
                output_file.write(
                    '(' + str(event[0]) + ', ' + str(event[1]) + ', ' + str(event[2]) + ', ' + str(event[3]) + ')\n')


if __name__ == "__main__":
    main()
