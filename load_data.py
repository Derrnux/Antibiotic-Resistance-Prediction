import random
import sys

class SEQData:
    def __init__(self, seed):
        random.seed(seed)
        # make sure that "/seminar-dlmb-2024-winter-public" exists

        sys.path.insert(0, 'seminar-dlmb-2024-winter-public/src/')

        from amr.amr_utility import load_gene_data, create_gene_datasets

        create_gene_datasets("seminar-dlmb-2024-winter-public/", "seminar-dlmb-2024-winter-public/data/ds1")
        ds = load_gene_data("seminar-dlmb-2024-winter-public/data/ds1", "Staphylococcus_aureus_cefoxitin", "pbp4")

        sequences_train = [x[1] for x in ds["train"]]
        labels_train = [x[2] for x in ds["train"]]

        sequences_test = [x[1] for x in ds["test"]]
        labels_test = [x[2] for x in ds["test"]]

        sequences = sequences_train + sequences_test
        labels = labels_train + labels_test

        class_0 = [(seq, label) for seq, label in zip(sequences, labels) if label == 0]
        class_1 = [(seq, label) for seq, label in zip(sequences, labels) if label == 1]

        random.shuffle(class_0)
        random.shuffle(class_1)

        test_data = class_0[:7] + class_1[:7]
        leftover_class_0 = class_0[7:]
        leftover_class_1 = class_1[7:]

        val_data = leftover_class_0[:7] + leftover_class_1[:7]
        leftover_class_0 = leftover_class_0[7:]
        leftover_class_1 = leftover_class_1[7:]

        train_data = leftover_class_0 + leftover_class_1

        self.test_sequences = [seq for seq, label in test_data]
        self.test_labels = [label for seq, label in test_data]

        self.val_sequences = [seq for seq, label in val_data]
        self.val_labels = [label for seq, label in val_data]

        self.train_sequences = [seq for seq, label in train_data]
        self.train_labels = [label for seq, label in train_data]