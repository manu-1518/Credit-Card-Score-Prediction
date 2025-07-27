import pandas as pd

def load_german_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = ['Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount',
               'Savings', 'Employment', 'InstallmentRate', 'PersonalStatusSex',
               'OtherDebtors', 'ResidenceSince', 'Property', 'Age', 'OtherInstallmentPlans',
               'Housing', 'ExistingCredits', 'Job', 'NumDependents', 'OwnTelephone', 'ForeignWorker', 'Target']
    df = pd.read_csv(url, delim_whitespace=True, header=None, names=columns)
    df['Target'] = df['Target'].map({1: 0, 2: 1})  # 0: good, 1: bad
    return df
