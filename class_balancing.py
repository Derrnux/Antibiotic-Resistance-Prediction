import random

class Balancing:
    def __init__(self):
        random.seed(42)

    def oversample_data(self, sequences, labels):
        class_0 = [(seq, label) for seq, label in zip(sequences, labels) if label == 0]
        class_1 = [(seq, label) for seq, label in zip(sequences, labels) if label == 1]

        # Oversample class 0
        num_class_1 = len(class_1)
        oversampled_class_0 = random.choices(class_0, k=num_class_1)

        balanced_data = oversampled_class_0 + class_1
        random.shuffle(balanced_data)

        balanced_sequences = [seq for seq, label in balanced_data]
        balanced_labels = [label for seq, label in balanced_data]

        return balanced_sequences, balanced_labels

    def undersample_data(self, sequences, labels):
        class_0 = [(seq, label) for seq, label in zip(sequences, labels) if label == 0]
        class_1 = [(seq, label) for seq, label in zip(sequences, labels) if label == 1]

        num_to_remove = len(class_1) - len(class_0)

        # Undersample class 1
        if num_to_remove > 0:
            class_1 = random.sample(class_1, len(class_1) - num_to_remove)

        balanced_data = class_0 + class_1
        random.shuffle(balanced_data)

        balanced_sequences = [seq for seq, label in balanced_data]
        balanced_labels = [label for seq, label in balanced_data]

        return balanced_sequences, balanced_labels