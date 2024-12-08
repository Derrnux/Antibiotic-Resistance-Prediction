import sys

class SEQData:
    def __init__(self):
        # make sure that "/seminar-dlmb-2024-winter-public" exists

        sys.path.insert(0, 'seminar-dlmb-2024-winter-public/src/')

        from amr.amr_utility import load_gene_data, create_gene_datasets

        create_gene_datasets("seminar-dlmb-2024-winter-public/", "seminar-dlmb-2024-winter-public/data/ds1")
        ds = load_gene_data("seminar-dlmb-2024-winter-public/data/ds1", "Staphylococcus_aureus_cefoxitin", "pbp4")

        self.seq_train = [x[1] for x in ds["train"]]
        self.y_train = [x[2] for x in ds["train"]]

        self.seq_test = [x[1] for x in ds["test"]]
        self.y_test = [x[2] for x in ds["test"]]
