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

# Remove the 'tag' field from all transactions except the first one
def remove_tag_except_first(accounts):
    for address, transactions in accounts.items():
        for i in range(1, len(transactions)):
            if 'tag' in transactions[i]:
                del transactions[i]['tag']

# Merge all transactions of each account into a single entry
def merge_transactions(accounts):
    for address in accounts.keys():
        if accounts[address]:
            first_tag = accounts[address][0]['tag']  # Keep the 'tag' from the first transaction
            merged_data = {'tag': first_tag, 'transactions': accounts[address]}
            accounts[address] = [merged_data]

# Load the data
accounts_data = load_data('transactions6.pkl')

# Remove redundant 'tag' fields
remove_tag_except_first(accounts_data)

# Merge all transactions per account into one record
merge_transactions(accounts_data)

# Save the processed data
save_data(accounts_data, 'transactions7.pkl')

# Print the first 10 processed records for each of the first 10 accounts
print("Printing the first 10 processed transactions for each of the first 10 accounts:")
for address in list(accounts_data.keys())[:10]:  # Show only the first 10 accounts
    print(f"First 10 transactions for account {address}:")
    for transaction in accounts_data[address][:10]:  # Show first 10 transactions per account
        print(transaction)
    print("\n")

print("Transaction data has been processed and saved to transactions7.pkl.")
