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

# Convert transaction data into descriptive text
def convert_transactions_to_text(accounts):
    for address, transactions in accounts.items():
        for idx, transaction in enumerate(transactions):
            tag = transaction['tag']
            transaction_descriptions = []
            for sub_transaction in transaction['transactions']:
                # Build a textual description of a single transaction
                description = ' '.join([f"{key}: {sub_transaction[key]}" for key in sub_transaction])
                transaction_descriptions.append(description)
            # Convert the transaction entry into a single line of text
            transactions[idx] = f"{tag} {'  '.join(transaction_descriptions)}."

# Load the data
accounts_data = load_data('transactions9.pkl')

# Convert transaction data into descriptive text
convert_transactions_to_text(accounts_data)

# Save the result
save_data(accounts_data, 'transactions10.pkl')

# Print the first 10 accounts' data
print("Printing the first 10 accounts:")
for address, transactions in list(accounts_data.items())[:10]:  # Show only the first 10 accounts
    print(f"Account {address}:")
    for transaction in transactions:
        print(transaction)
    print("\n")

print("Data has been converted into descriptive text and saved to transactions10.pkl.")
