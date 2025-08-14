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

# Remove specific fields from each transaction
def remove_fields(accounts, fields):
    for address in tqdm.tqdm(accounts.keys(), desc="Removing fields"):
        for transaction in accounts[address]:
            for field in fields:
                if field in transaction:
                    del transaction[field]

# Load the input data
accounts_data = load_data('transactions4.pkl')

# Define which fields to remove
fields_to_remove = ['from_address', 'to_address', 'timestamp']

# Remove the specified fields
remove_fields(accounts_data, fields_to_remove)

# Save the cleaned data
save_data(accounts_data, 'transactions5.pkl')

# Print the first 10 processed transactions for the first 10 accounts
print("Printing the first 10 processed transactions for each of the first 10 accounts:")
for address in list(accounts_data.keys())[:10]:  # Show only the first 10 accounts
    print(f"First 10 transactions for account {address}:")
    for transaction in accounts_data[address][:10]:  # Show the first 10 transactions per account
        print(transaction)
    print("\n")

print("Fields have been removed and data saved to transactions5.pkl.")
