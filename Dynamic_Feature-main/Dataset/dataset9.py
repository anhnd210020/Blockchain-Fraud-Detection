import pickle
import tqdm

# Load data from a file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to a file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Remove the 'tag' field from each transaction inside the 'transactions' list
def remove_tag_from_transactions(accounts):
    for address, transactions in accounts.items():
        for transaction in transactions:
            for sub_transaction in transaction['transactions']:
                if 'tag' in sub_transaction:
                    del sub_transaction['tag']

# Load the data
accounts_data = load_data('transactions8.pkl')

# Remove the 'tag' field from nested transactions
remove_tag_from_transactions(accounts_data)

# Save the processed data
save_data(accounts_data, 'transactions9.pkl')

# Print the first 10 accounts' data
print("Printing data for the first 10 accounts:")
for address, transactions in list(accounts_data.items())[:10]:  # Show only the first 10 accounts
    print(f"Account {address}:")
    for transaction in transactions:
        print(transaction)
    print("\n")

print("The 'tag' field has been removed and the data has been saved to transactions9.pkl.")
