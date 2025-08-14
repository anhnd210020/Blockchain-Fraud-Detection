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

# Select and shuffle accounts
def select_and_shuffle_accounts(accounts):
    tag1_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 1]
    tag0_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 0]
    
    # Randomly select tag=0 accounts with a quantity twice the number of tag=1 accounts
    double_tag1_count = random.sample(tag0_accounts, 2 * len(tag1_accounts))
    
    # Combine and shuffle
    selected_accounts = tag1_accounts + double_tag1_count
    random.shuffle(selected_accounts)
    
    # Return as a shuffled dictionary
    return dict(selected_accounts)

# Load data
accounts_data = load_data('transactions7.pkl')

# Select and shuffle accounts
shuffled_accounts_data = select_and_shuffle_accounts(accounts_data)

# Save the processed data
save_data(shuffled_accounts_data, 'transactions8.pkl')

# Print the first 10 processed accounts
print("Printing the first 10 accounts:")
for address, transactions in list(shuffled_accounts_data.items())[:10]:  # Show only the first 10 accounts
    print(f"Account {address}:")
    print(transactions)
    print("\n")

print("Data has been processed and saved to transactions8.pkl.")
