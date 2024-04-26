from query.random import random_query
from query.margin import margin_query
from query.badge import badge_query
from query.coreset import coreset_query
from query.typiclust import typiclust_query
from query.bait import bait_query


def build_query(query_strategy):
    if query_strategy == 'random':
        return random_query
    elif query_strategy == 'margin':
        return margin_query
    elif query_strategy == 'coreset':
        return coreset_query
    elif query_strategy == 'badge':
        return badge_query
    elif query_strategy == 'typiclust':
        return typiclust_query
#    elif query_strategy == 'bait':
#        return bait_query
    else:
        raise NotImplementedError