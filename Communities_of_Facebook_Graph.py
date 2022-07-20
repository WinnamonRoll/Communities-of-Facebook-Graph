#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:20:24 2022

@author: winnamonroll
"""

import os
import random
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def check_disjoint(communities):
    for community1 in communities.values():
        for community2 in communities.values():
            if (community1 is not community2) and (not community1.isdisjoint(community2)):
                return False
    return True
    
def get_communities(attribute):
    communities = dict()
    communities["N/A"] = set()
    
    files_numbers = list()
    for filename in os.listdir("./facebook/"):
        if filename.split('.')[0] not in files_numbers:
            files_numbers.append(int(filename.split('.')[0]))
            
    for file_number in files_numbers:
        featnames_file = open("./facebook/" + str(file_number) + ".featnames", "r")
        features_names = list()
        for line in featnames_file.readlines():
            name = (line.split())[1].replace(';', '_')
            feature_id = int((line.split())[-1])
            features_names.append((name, feature_id))
        
        ego_feat_file = open("./facebook/" + str(file_number) + ".egofeat", "r")
        ego_node_data = [int(x) for x in str.split(ego_feat_file.readline())]
        ego_node = file_number
        ego_node_features = ego_node_data[1:]
        not_given = True
        for i in range(len(ego_node_features)):
            if ego_node_features[i] == 1 and features_names[i][0] == attribute:
                if (features_names[i][1] not in communities):
                    communities[features_names[i][1]] = set()
                communities[features_names[i][1]].add(ego_node)
                not_given = False
            if not_given:
                communities["N/A"].add(ego_node)
                
        feat_file = open("./facebook/" + str(file_number) + ".feat", "r")
        feat_file_lines = feat_file.readlines()
        num_feat_file_lines = len(feat_file_lines)
        for node in range(num_feat_file_lines):
            node_data = [int(x) for x in str.split(feat_file_lines[node])]
            node = node_data[0]
            node_features = node_data[1:]
            not_given = True
            for i in range(len(node_features)):
                if node_features[i] == 1 and features_names[i][0] == attribute:
                    if (features_names[i][1] not in communities):
                        communities[features_names[i][1]] = set()
                    communities[features_names[i][1]].add(node)
                    not_given = False
            if not_given:
                communities["N/A"].add(node)
    
        ego_feat_file.close()
        feat_file.close()
        featnames_file.close()
        
    return communities


def get_inverse_communities(communities):
    inverse_communities = dict()
    for community in communities:
        for node in communities[community]:
            if node not in inverse_communities:
                inverse_communities[node] = set()
            inverse_communities[node].add(community)
        
    return inverse_communities


def get_disjoint_communities(inverse_communities):
    disjoint_communities = dict()
    disjoint_communities["N/A"] = set()
    for node in inverse_communities:
        if len(inverse_communities[node]) == 1:
            community = tuple(inverse_communities[node])[0]
            if community not in disjoint_communities:
                disjoint_communities[community] = set()
            disjoint_communities[community].add(node)
        elif len(inverse_communities[node]) > 1:
            for community in tuple(inverse_communities[node]):
                if community not in disjoint_communities:
                    disjoint_communities[community] = set()
            community = random.choice(tuple(inverse_communities[node]))
            disjoint_communities[community].add(node)
    
    return disjoint_communities


def get_disjoint_inverse_communities(disjoint_communities):
    disjoint_inverse_communities = dict()
    for community in disjoint_communities:
        for node in tuple(disjoint_communities[community]):
            disjoint_inverse_communities[node] = set()
            disjoint_inverse_communities[node].add(community)
    return disjoint_inverse_communities


def compute_communities_adjacency_matrix(G, inverse_communities, \
    communities_to_indices, Normalized = True):
    
    num_communities = len(communities_to_indices)
    communities_adjacency_matrix = np.zeros((num_communities, num_communities))
    for edge in G.edges:
        for community_i in inverse_communities[edge[0]]:
            for community_j in inverse_communities[edge[1]]:
                communities_adjacency_matrix[ \
                    communities_to_indices[community_i], \
                    communities_to_indices[community_j]] += 1
                communities_adjacency_matrix[ \
                    communities_to_indices[community_j], \
                    communities_to_indices[community_i]] += 1
    # print(np.sum(communities_adjacency_matrix)//2)
    if Normalized:
        row_sums = np.sum(communities_adjacency_matrix, axis=0)
        column_sums = np.sum(communities_adjacency_matrix, axis=1)
        communities_adjacency_matrix /= np.outer(row_sums, column_sums)
        
    # to handle division by zero
    communities_adjacency_matrix = np.nan_to_num(communities_adjacency_matrix)
        
    return communities_adjacency_matrix


def sort_matrix_by_ids(matrix, communities_to_indices):
    sorted_communities_to_indices = [(label, index) for label, index in \
        communities_to_indices if label != "N/A"]
    sorted_communities_to_indices = sorted(sorted_communities_to_indices, \
        key=lambda x:x[0])
    for community, index in communities_to_indices:
        if community == "N/A":
            sorted_communities_to_indices.insert(0, (community, index))
    clusters = [index for label, index in sorted_communities_to_indices]
    sorted_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        sorted_matrix[i, :] = matrix[clusters[i], :][clusters]
    return (sorted_matrix, sorted_communities_to_indices)


def plot_matrix(matrix, labels_to_indices, title, plot_log = False):
    fig, ax = plt.subplots(figsize=(32, 32))
    if plot_log:
        im = ax.imshow(np.log(matrix+np.min(matrix[matrix > 0])), cmap=plt.cm.jet)
    else:
        im = ax.imshow(matrix, cmap=plt.cm.jet)

    labels = [labels for labels, index in labels_to_indices]
    ax.set_title(title, {'fontsize': 64})
    ax.set_xticks(np.arange(matrix.shape[1]), \
        labels=labels)
    ax.set_yticks(np.arange(matrix.shape[0]), \
        labels=labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, \
        labelsize=2000/matrix.shape[0], width=200/matrix.shape[0], \
        length=1600/matrix.shape[0])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="center", rotation_mode="default")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth = \
        50/matrix.shape[0])
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()
    

def simulate(G, attribute, num_iterations):
    print("Starting simulation on", attribute, "with", num_iterations, \
          "iterations.")
    communities = get_communities(attribute)
    inverse_communities = get_inverse_communities(communities)
    communities_to_indices = {community : i \
          for i, community in enumerate(communities.keys())}
    mean_communities_adjacency_matrix = None
    mean_modularity = None
    for i in range(num_iterations):
        print("attribute :", attribute, "---", "iteration", str(i))
        disjoint_communities = get_disjoint_communities(inverse_communities)
        disjoint_inverse_communities = \
            get_disjoint_inverse_communities(disjoint_communities)
        if i == 0:
            mean_communities_adjacency_matrix = \
                compute_communities_adjacency_matrix(G, \
                disjoint_inverse_communities, communities_to_indices, \
                Normalized = True)
            mean_modularity = nx_comm.modularity(G, disjoint_communities.values())
        else:
            new_communities_adjacency_matrix = \
                compute_communities_adjacency_matrix(G, \
                disjoint_inverse_communities, communities_to_indices, \
                Normalized = True)
            mean_communities_adjacency_matrix += \
                new_communities_adjacency_matrix
            mean_modularity += \
                nx_comm.modularity(G, disjoint_communities.values())
            
    mean_communities_adjacency_matrix /= num_iterations
    mean_modularity = mean_modularity/num_iterations
    
    mean_communities_adjacency_matrix, labels_to_indices = \
        sort_matrix_by_ids(mean_communities_adjacency_matrix, \
        list(communities_to_indices.items()))
    
    plot_matrix(mean_communities_adjacency_matrix, labels_to_indices, \
        "mean " + attribute + " adjacency matrix")
    plot_matrix(mean_communities_adjacency_matrix, labels_to_indices, \
        "log of mean " + attribute + " adjacency matrix", plot_log = True)
    print("=" * 80, end="\n\n")
    return (mean_communities_adjacency_matrix, mean_modularity)
            

# loading the nodes and edges
G = nx.read_edgelist("./facebook_combined.txt", create_using = nx.Graph(), \
    nodetype = int)
attributes = ["work_employer_id_anonymized", "work_position_id_anonymized", \
              "work_location_id_anonymized", "work_start_date_anonymized", \
              "work_end_date_anonymized"]
num_iterations = 1000

modularities = list()
for attribute in attributes:
    communities_adjacency_matrix, modularity = \
        simulate(G, attribute, num_iterations)
    modularities.append(modularity)

fig = plt.figure()
plt.bar(attributes, modularities)
plt.title("Mean Modularity of each attribute")
plt.ylabel("mean modularity")
plt.xticks(rotation = 60, ha="right", fontsize=8)
plt.show()
    


