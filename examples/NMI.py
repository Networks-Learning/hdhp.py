from collections import Counter
import math
import itertools

def calculate_all_NMIs(true_labels, estimated_labels, num_of_patterns):

    patterns_list = range(0, num_of_patterns)
    permutations = itertools.permutations(patterns_list)
    max_NMI = 0
    final_order = []

    for current_permutation in permutations:
        new_labels = estimated_labels[:]

        for j in range(len(new_labels)):
            # print(str(new_labels[j]) + " ---> " + str(current_permutation[new_labels[j]]))
            new_labels[j] = current_permutation[new_labels[j]]

        NMI = calculate_single_NMI(true_labels, estimated_labels)
        print("NMI: " + str(NMI))

        if NMI > max_NMI:
            max_NMI = NMI
            final_order = current_permutation[:]

    print (max_NMI, final_order)


def calculate_single_NMI(true_labels, estimated_labels):

    class_probs = {}
    cluster_probs = {}
    conditional_probs = {}

    for i in range(len(true_labels)):

        true_label = true_labels[i]
        true_labels.append(true_label)
        if true_label in class_probs:
            class_probs[true_label] += 1
        else:
            class_probs[true_label] = 1

        pred_label = estimated_labels[i]
        estimated_labels.append(pred_label)
        if pred_label in cluster_probs:
            cluster_probs[pred_label] += 1
        else:
            cluster_probs[pred_label] = 1

        if pred_label in conditional_probs:
            if true_label in conditional_probs.get(pred_label):
                conditional_probs.get(pred_label)[true_label] += 1
            else:
                conditional_probs.get(pred_label)[true_label] = 1
        else:
            conditional_probs[pred_label] = {true_label: 1}

    for cluster_label in conditional_probs:
        for class_label in conditional_probs.get(cluster_label):
            conditional_probs.get(cluster_label)[class_label] /= cluster_probs.get(cluster_label) * 1.0

    conditional_entropy = 0

    for cluster_label in conditional_probs:
        for class_label in conditional_probs.get(cluster_label):
            conditional_entropy += (-conditional_probs.get(cluster_label).get(class_label) * math.log(
                conditional_probs.get(cluster_label).get(class_label), 2))

    for key in class_probs:
        class_probs[key] /= len(true_labels) * 1.0
    class_labels_entropy = 0

    for key in class_probs:
        class_labels_entropy += (-class_probs.get(key) * math.log(class_probs.get(key), 2))

    for key in cluster_probs:
        cluster_probs[key] /= len(estimated_labels) * 1.0

    cluster_labels_entropy = 0

    for key in cluster_probs:
        cluster_labels_entropy += (-cluster_probs.get(key) * math.log(cluster_probs.get(key), 2))

    IC = class_labels_entropy - conditional_entropy

    NMI = (2 * IC) / (class_labels_entropy + cluster_labels_entropy)

    print("NMI: " + str(NMI))
    return NMI

#
# file_path = "results/1/U_40_E_10000_patterns.tsv"
#
# true_labels = []
# estimated_labels = []
#
# lines = open(file_path).readlines()
# class_probs = {}
# cluster_probs = {}
# conditional_probs = {}
#
# for line in lines:
#     splitted_line = line.split('\t')
#     true_label = splitted_line[0].strip()
#     true_labels.append(true_label)
#     if true_label in class_probs:
#         class_probs[true_label] += 1
#     else:
#         class_probs[true_label] = 1
#
#     pred_label = splitted_line[1].strip()
#     estimated_labels.append(pred_label)
#     if pred_label in cluster_probs:
#         cluster_probs[pred_label] += 1
#     else:
#         cluster_probs[pred_label] = 1
#
#     if pred_label in conditional_probs:
#         if true_label in conditional_probs.get(pred_label):
#             conditional_probs.get(pred_label)[true_label] += 1
#         else:
#             conditional_probs.get(pred_label)[true_label] = 1
#     else:
#         conditional_probs[pred_label] = {true_label: 1}
#
#
# for cluster_label in conditional_probs:
#     for class_label in conditional_probs.get(cluster_label):
#         conditional_probs.get(cluster_label)[class_label] /= cluster_probs.get(cluster_label) * 1.0
#
# conditional_entropy = 0
#
# for cluster_label in conditional_probs:
#     for class_label in conditional_probs.get(cluster_label):
#         conditional_entropy += (-conditional_probs.get(cluster_label).get(class_label) * math.log(conditional_probs.get(cluster_label).get(class_label), 2))
#
#
# for key in class_probs:
#     class_probs[key] /= len(true_labels) * 1.0
# class_labels_entropy = 0
#
# for key in class_probs:
#     class_labels_entropy += (-class_probs.get(key) * math.log(class_probs.get(key), 2))
#
#
# for key in cluster_probs:
#     cluster_probs[key] /= len(estimated_labels) * 1.0
#
# cluster_labels_entropy = 0
#
# for key in cluster_probs:
#     cluster_labels_entropy += (-cluster_probs.get(key) * math.log(cluster_probs.get(key), 2))
#
#
# IC = class_labels_entropy - conditional_entropy
#
# NMI = (2 * IC) / (class_labels_entropy + cluster_labels_entropy)
#
# print("NMI: " + str(NMI))



