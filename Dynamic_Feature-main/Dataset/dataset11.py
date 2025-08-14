import pickle
from sklearn.model_selection import train_test_split

# Load data from the transactions10.pkl file
with open('transactions10.pkl', 'rb') as file:
    transactions_dict = pickle.load(file)

# Convert the dictionary into a list, where each element is a "tag sentence" string
transactions = []
for key, value_list in transactions_dict.items():
    for value in value_list:
        transactions.append(f"{value}")  # Assume key is tag, value is sentence

# Define the split ratios
train_size = 0.8
validation_size = 0.1
test_size = 0.1

# First split into training and remaining sets
train_data, temp_data = train_test_split(transactions, train_size=train_size, random_state=42)

# Then split remaining into validation and test sets
validation_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + validation_size), random_state=42)

# Save train and validation sets to TSV format
def save_to_tsv_train_dev(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("label\tsentence\n")
        for line in data:
            tag, sentence = line.split(' ', 1) # Split the tag from the sentence
            file.write(f"{tag}\t{sentence}\n")

# Split the tag from the sentence
def save_to_tsv_test(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("index\tsentence\n")
        for idx, line in enumerate(data):
            tag, sentence = line.split(' ', 1)
            file.write(f"{idx}\t{sentence}\n")

# Save all splits
save_to_tsv_train_dev(train_data, 'train.tsv')
save_to_tsv_train_dev(validation_data, 'dev.tsv')
save_to_tsv_test(test_data, 'test.tsv')

print("Files saved: train.tsv, dev.tsv, test.tsv")
