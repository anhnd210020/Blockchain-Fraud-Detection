import pickle

# Load data from a file
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Save data to a file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Main processing function
def process_transactions(transactions):
    # Create a dictionary to store transactions for each address
    accounts = {}

    # Process each transaction
    for tx in transactions:
        # Add "outgoing" transaction for sender
        from_address = tx['from_address']
        if from_address not in accounts:
            accounts[from_address] = []
        accounts[from_address].append({**tx, 'in_out': 1})  # 添加转出标志

        # Add "incoming" transaction for receiver
        to_address = tx['to_address']
        if to_address not in accounts:
            accounts[to_address] = []
        accounts[to_address].append({**tx, 'in_out': 0})  # 添加转入标志

    return accounts

# Load transaction data
transactions = load_data('transactions1.pkl')

# Process and organize data per account
processed_data = process_transactions(transactions)

# Save processed data to a new file
save_data(processed_data, 'transactions2.pkl')

# Print the first five transactions for the first ten accounts for inspection
for address in list(processed_data.keys())[:10]:  # Show only the first 10 accounts
    print(f"Transactions for account {address}:")
    for transaction in processed_data[address][:5]:  # Show first 5 transactions per account
        print(transaction)
    print("\n")
