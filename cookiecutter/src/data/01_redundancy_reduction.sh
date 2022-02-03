#!/bin/bash

mmseqs easy-cluster ../../data/raw/antigens_before_clustering.fasta ../../data/interim/Id50ClusterRes ../../data/interim/mmseqs2_tmp --min-seq-id 0.5 -c 0.8 --cov-mode 1
cut -f1 ../../data/interim/Id50ClusterRes_cluster.tsv | cut -f1 | uniq > ../../data/interim/non_redundant_chain_ids.tst.txt 

