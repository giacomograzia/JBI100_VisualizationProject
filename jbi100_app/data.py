import pandas as pd


def get_data():
    """
    This function cleans and preprocesses all the data from the initial all_data.csv file from kaggle.
    The final output creates the final_credit_0_60 dataframe
    :return: cleaned dataframe
    """
    types = {"Customer_ID": str}
    # Read data
    df = pd.read_csv('all_data.csv', sep=';', low_memory=False, dtype=types)

    # columns we need
    df = df[['Customer_ID', 'Age', 'Month', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card',
             'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly']]

    def only_numerics(seq):
        """
        Remove all non-numeric characters from a given sequence and return a sequence of the
        same type containing only numeric characters
        :param seq: sequence to be operated on
        :return: sequence with only numeric characters
        """
        seq_type = type(seq)
        return seq_type().join(filter(seq_type.isdigit, seq))

    # Age - correct types
    df.Age = df.Age.apply(lambda x: only_numerics(x)).astype('int')
    # set type to integer
    df.Age = df.Age.astype('int')

    # Annual income - correct types
    df.Annual_Income = df.Annual_Income.str.replace(',', '.')
    df.Annual_Income = df.Annual_Income.str.strip('_')
    df.Annual_Income = df.Annual_Income.astype('float')

    # Monthly Inhand Salary - correct types
    df.Monthly_Inhand_Salary = df.Monthly_Inhand_Salary.str.replace(',', '.')
    df.Monthly_Inhand_Salary = df.Monthly_Inhand_Salary.astype('float')

    # Total_EMI_per_month - correct types
    df.Total_EMI_per_month = df.Total_EMI_per_month.str.replace(',', '.')
    df.Total_EMI_per_month = df.Total_EMI_per_month.astype('float')

    # Amount_invested_monthly - correct types
    df.Amount_invested_monthly = df.Amount_invested_monthly.str.replace(',', '.')
    df.Amount_invested_monthly = df.Amount_invested_monthly.str.strip('__')
    df.Amount_invested_monthly = df.Amount_invested_monthly.astype('float')

    # Outstanding Debt - correct types
    df.Outstanding_Debt = df.Outstanding_Debt.str.replace(',', '.')
    df.Outstanding_Debt = df.Outstanding_Debt.str.strip('_')
    df.Outstanding_Debt = df.Outstanding_Debt.astype('float')

    # Occupation
    df.Occupation = df.Occupation.replace('_______', None)

    # in some cases, the occupation is empty, in this case we try to get it from another row since
    # users have multiple rows
    occupation_dict = dict()

    for index, row in df.iterrows():
        id_ = row['Customer_ID']
        occupation = row['Occupation']

        # Check if the customer ID is already in the dictionary
        if id_ in occupation_dict:
            pass
        else:
            if occupation != None:
                occupation_dict[id_] = occupation
            else:
                pass

    x = pd.Series(occupation_dict)

    def replace_missing_occupation(row):
        """
        To fix some None occupation entries, use apply function to all rows.
        Function looks up the customer id in the dictionary and of occupations and fills in occupation
        entry with ocupation, if present.
        :param row: the row in the dataframe
        :return: the occupation
        """
        id_ = row['Customer_ID']
        occupation = row['Occupation']

        # Check if the occupation is missing
        if pd.isnull(occupation) and id_ in occupation_dict:
            return occupation_dict[id_]

        return occupation

    # Apply the custom function to replace missing occupations
    df['Occupation'] = df.apply(replace_missing_occupation, axis=1)

    # in some cases, the monthly inhand salary is empty, in this case we try to get it from another row since
    # users have multiple rows
    monthly_inhand_salary_dict = dict()

    for index, row in df.iterrows():
        id_ = row['Customer_ID']
        monthly_inhand_salary = row['Monthly_Inhand_Salary']

        # Check if the customer ID is already in the dictionary
        if id_ in monthly_inhand_salary_dict:
            pass
        else:
            if monthly_inhand_salary != None:
                monthly_inhand_salary_dict[id_] = monthly_inhand_salary
            else:
                pass

    # in some cases, the amount of the monthly investment is empty, in this case we try to get it from another row since
    # users have multiple rows
    amount_invested_monthly_dict = dict()

    for index, row in df.iterrows():
        id_ = row['Customer_ID']
        amount_invested_monthly = row['Amount_invested_monthly']

        # Check if the customer ID is already in the dictionary
        if id_ in amount_invested_monthly_dict:
            pass
        else:
            if amount_invested_monthly != None:
                amount_invested_monthly_dict[id_] = amount_invested_monthly
            else:
                pass

    def replace_missing_value(row, replace_entry, dict_):
        """
        Rplacing missing values by finding the user in a different row and taking the value from there
        :param row: row in the dataframe
        :param replace_entry: item that needs to be replaces
        :param dict_: dictionary
        :return: the new value
        """
        id_ = row['Customer_ID']
        val = row[replace_entry]

        # Check if the occupation is missing
        if pd.isnull(val) and id_ in dict_:
            return dict_[id_]

        return val

    df['Amount_invested_monthly'] = df.apply(
        lambda x: replace_missing_value(x, 'Amount_invested_monthly', amount_invested_monthly_dict), axis=1)

    # Monthly_Inhand_Salary (average) keep one value per customer
    avg_df = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].mean().reset_index()

    # Total_EMI_per_month (average)
    avg_total_emi_per_month = df.groupby('Customer_ID')['Total_EMI_per_month'].mean().reset_index()
    avg_df = avg_df.merge(avg_total_emi_per_month)

    # Annual_Income (average)
    avg_annual_income = df.groupby('Customer_ID')['Annual_Income'].mean().reset_index()
    avg_df = avg_df.merge(avg_annual_income)

    # Amount_invested_monthly (average)
    avg_amount_invested_monthly = df.groupby('Customer_ID')['Amount_invested_monthly'].mean().reset_index()
    avg_df = avg_df.merge(avg_amount_invested_monthly)

    # in some cases, the occupation is empty, in this case we try to get it from another row since
    # users have multiple rows
    occupation_dict = dict()

    for index, row in df.iterrows():
        id_ = row['Customer_ID']
        occupation = row['Occupation']

        # Check if the customer ID is already in the dictionary
        if id_ in occupation_dict:
            pass
        else:
            if occupation != None:
                occupation_dict[id_] = occupation
            else:
                pass

    # in some cases, the age is empty, in this case we try to get it from another row since
    # users have multiple rows
    age_dict = dict()

    for index, row in df.iterrows():
        id_ = row['Customer_ID']
        age = row['Age']
        occupation = row['Occupation']

        if id_ in age_dict:
            pass
        else:
            if age != None:
                age_dict[id_] = age
            else:
                pass

    age_df = pd.DataFrame(age_dict.items(), columns=['Customer_ID', 'Age'])

    occupation_df = pd.DataFrame(occupation_dict.items(), columns=['Customer_ID', 'Occupation'])

    age_occupation_df = age_df.merge(occupation_df)

    final_df = age_occupation_df.merge(avg_df)

    # getting rid of unrealistic ages
    final_df_0_60 = final_df[final_df['Age'] < 60]

    # uncomment if you want to get the csv as well
    # final_df_0_60.to_csv('jbi100_app/assets/final_credit_0_60.csv')

    return final_df_0_60


final_df_0_60 = get_data()

