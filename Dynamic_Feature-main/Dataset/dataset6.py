import pickle
import random
import tqdm

# Load data from a file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to a file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Shuffle the order of transactions for each account
def shuffle_transactions(accounts):
    for address in tqdm.tqdm(accounts.keys(), desc="Shuffling transaction order"):
        random.shuffle(accounts[address])

# Load the cleaned transaction data
accounts_data = load_data('transactions5.pkl')

# Shuffle the transaction sequences
shuffle_transactions(accounts_data)

# Save the shuffled data
save_data(accounts_data, 'transactions6.pkl')

# Print the first 5 shuffled transactions for the first 5 accounts
print("Printing the first 5 shuffled transactions for each of the first 5 accounts:")
for address in list(accounts_data.keys())[:5]:  # Show only the first 5 accounts
    print(f"First 5 transactions for account {address}:")
    for transaction in accounts_data[address][:5]: # Show the first 5 transactions per account
        print(transaction)
    print("\n")

print("Transaction data has been shuffled and saved to transactions6.pkl.")
