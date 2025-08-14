import pickle

# Load data from a file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to a file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Sort the transactions of each account by timestamp
def sort_transactions_by_timestamp(accounts):
    sorted_accounts = {}
    for address, transactions in accounts.items():
        sorted_accounts[address] = sorted(transactions, key=lambda x: x['timestamp'])
    return sorted_accounts

# Load the per-account transaction data
accounts_data = load_data('transactions2.pkl')

# Sort transactions by timestamp for each account
sorted_accounts_data = sort_transactions_by_timestamp(accounts_data)

# Print the first 10 sorted transactions for each of the first 10 accounts
print("Printing the first 10 sorted transactions for each of the first 10 accounts:")
for address in list(sorted_accounts_data.keys())[:10]:  # Show only the first 10 accounts
    print(f"First 10 transactions for account {address}:")
    for transaction in sorted_accounts_data[address][:10]:  # Show first 10 transactions per account
        print(transaction)
    print("\n")

# Save the sorted data
save_data(sorted_accounts_data, 'transactions3.pkl')

print("Transactions have been sorted by timestamp per account and saved to transactions3.pkl.")
