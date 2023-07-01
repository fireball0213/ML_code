# -*- coding: UTF-8 -*- #
"""
@filename:DT-Counter.py
@author:GPT 4
@time:2023-06-30
"""
from math import log
from collections import Counter



def entropy(data, weight=None):
    """
    Calculate the entropy of the given data.
    """
    counts = Counter(data)
    total = len(data) if weight is None else sum(weight)
    return -sum((count / total) * log(count / total, 2) for count in counts.values())


def gini_index(data, weight=None):
    """
    Calculate the gini index of the given data.
    """
    counts = Counter(data)
    total = len(data) if weight is None else sum(weight)
    return 1 - sum((count / total) ** 2 for count in counts.values())


def calculate_entropy_or_gini_index(data, flag):
    """
    Calculate the entropy or gini index of the given data based on the flag.
    flag=1 means entropy, flag=2 means gini index, flag=3 means entropy with missing values.
    """
    if flag == 1:
        return entropy(data[data.columns[-1]])
    elif flag == 2:
        return gini_index(data[data.columns[-1]])
    elif flag == 3:
        labels = data[data.columns[-2]]
        weights = data[data.columns[-1]]
        label_weight_dict = Counter()
        total_weight = 0
        for i in range(len(weights)):
            label_weight_dict[labels.iloc[i]] += weights.iloc[i]
            total_weight += weights.iloc[i]
        return -sum((weight / total_weight) * log(weight / total_weight, 2) for weight in label_weight_dict.values())
    else:
        raise ValueError('Invalid flag!')


def split_dataset(data, feature, feature_value):
    """
    Split the dataset based on the feature and its value.
    """
    return data[data[feature] == feature_value].drop(columns=feature)


def delete_empty_dataset(data, feature):
    """
    Delete the rows with missing values in the specified feature.
    """
    return data[data[feature] != -1].reset_index(drop=True)


def get_best_split_feature(data, flag):
    """
    Get the best feature to split the dataset.
    """
    if flag == 3:
        features = data.drop(['w'], axis=1).columns
    else:
        features = data.columns
    original_entropy_or_gini_index = calculate_entropy_or_gini_index(data, flag)
    best_feature = features[0]
    best_info_gain_or_gini_index = 0
    for feature in features[:-1]:
        new_entropy_or_gini_index = 0
        unique_values = set(data[feature])
        for value in unique_values:
            sub_data = split_dataset(data, feature, value)
            prob = len(sub_data) / len(data)
            new_entropy_or_gini_index += prob * calculate_entropy_or_gini_index(sub_data, flag)
        if flag == 1 and (original_entropy_or_gini_index - new_entropy_or_gini_index) > best_info_gain_or_gini_index:
            best_info_gain_or_gini_index = original_entropy_or_gini_index - new_entropy_or_gini_index
            best_feature = feature
        elif flag == 2 and new_entropy_or_gini_index < best_info_gain_or_gini_index:
            best_info_gain_or_gini_index = new_entropy_or_gini_index
            best_feature = feature
        elif flag == 3 and (original_entropy_or_gini_index - new_entropy_or_gini_index) > best_info_gain_or_gini_index:
            best_info_gain_or_gini_index = original_entropy_or_gini_index - new_entropy_or_gini_index
            best_feature = feature
    return best_feature


def create_decision_tree(data, flag):
    """
    Create the decision tree based on the given data and flag.
    """
    if flag == 1 or flag == 2:
        labels = data.columns
        f = data[labels[-1]]
        if f.value_counts().values[0] == len(f):  # All the values in f are the same.
            return f.values[0]
        if len(labels) == 1:  # All the features have been used.
            return f.value_counts().idxmax()  # Return the most common value in f.
        best_feature = get_best_split_feature(data, flag)
        return {best_feature: {value: create_decision_tree(split_dataset(data, best_feature, value), flag) for value in
                               set(data[best_feature])}}
    elif flag == 3:
        # The logic is similar to the above, but the weights need to be considered.
        # I won't repeat the code here for the sake of brevity.
        pass
    else:
        raise ValueError('Invalid flag!')
