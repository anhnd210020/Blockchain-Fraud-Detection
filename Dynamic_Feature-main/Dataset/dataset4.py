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

# Add n-gram data to each transaction
def add_n_grams(accounts):
    for address, transactions in tqdm.tqdm(accounts.items(), desc="Processing n-gram data"):
        for n in range(2, 6):  # Process 2-gram to 5-gram
            gram_key = f"{n}-gram"
            for i in range(len(transactions)):
                if i < n-1:
                    transactions[i][gram_key] = 0  # First n-1 transactions set to 0
                else:
                    transactions[i][gram_key] = transactions[i]['timestamp'] - transactions[i-n+1]['timestamp']

# Load the sorted per-account transaction data
accounts_data = load_data('transactions3.pkl')

# Add n-gram temporal features
add_n_grams(accounts_data)

# Save the updated data
save_data(accounts_data, 'transactions4.pkl')

# Print the first 10 processed transactions for the first 10 accounts
print("Printing the first 10 processed transactions for each of the first 10 accounts:")
for address in list(accounts_data.keys())[:10]: # Only show the first 10 accounts
    print(f"First 10 transactions for account {address}:")
    for transaction in accounts_data[address][:10]:  # Show the first 10 transactions per account
        print(transaction)
    print("\n")

print("n-gram features have been computed and saved to transactions4.pkl.")
