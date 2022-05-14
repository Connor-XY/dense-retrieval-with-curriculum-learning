#!/bin/bash

data_dir="/data"

mkdir -p "${data_dir}"
mkdir -p "models"
wget -O models/tfidf.pickle https://umd.box.com/s/uxf1gvh6lplh3fxge7huk396h8dcdei3
wget -O models/new_guesser/.../pytorch_model.bin xxx
wget -O models/new_guesser/.../config.json yyy
wget -O new_models/new_guesser/.../pytorch_model.bin aaa
wget -O new_models/new_guesser/.../config.json zzz


# Download the full train data
if [ ! -f "${data_dir}/qanta.train.2018.json" ]; then
    echo "Downloading qanta.train.2018.04.18.json as full train set."
    wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json"
    mv qanta.train.2018.json "${data_dir}/qanta.train.2018.json"
    echo "Download complete.\n"
fi

# Download the full dev data
if [ ! -f "${data_dir}/qanta.dev.2018.json" ]; then
    echo "Downloading qanta.dev.2018.04.18.json as full dev set."
    wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json"
    mv qanta.dev.2018.json "${data_dir}/qanta.dev.2018.json"
    echo "Download complete.\n"
fi

# Download the Wiki Look up jsons
if [ ! -f "${data_dir}/wiki_lookup.2018.json" ]; then
    echo "Downloading Wiki Lookup jsons 2018.04.18"
    wget "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/wikipedia/wiki_lookup.json"
    mv wiki_lookup.json "${data_dir}/wiki_lookup.2018.json"
    echo "Download complete.\n"
fi

if [ ! -f "${data_dir}/train-v2.0.json" ]; then
    echo "Downloading SQuAD train-v2.0.json"
    wget "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    mv train-v2.0.json "${data_dir}/train-v2.0.json"
    echo "Download complete.\n"
fi