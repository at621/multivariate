import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import scipy.stats as stats


# Gini Coefficient Calculation
def calculate_gini(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return 2 * auc - 1

def observed_expected_woe(df, feature, target, predicted):
    """
    Calculate the Weight of Evidence (WOE) for both observed and expected values.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    feature (str): The feature for which WOE is being calculated.
    target (str): The name of the actual target variable column.
    predicted (str): The name of the predicted target variable column.

    Returns:
    pandas.DataFrame: A combined dataframe with both observed and expected WOE.
    """

    def calculate_woe_stats(grouped, target, suffix):
        """
        Calculate WOE statistics for a given grouping.

        Parameters:
        grouped (DataFrameGroupBy): Grouped dataframe.
        prefix (str): Prefix to distinguish between observed and expected WoE
        """
        
        stats = grouped[target].agg(['count', 'sum'])
        stats.columns = [f'total_{suffix}', f'bad_{suffix}']

        stats[f'good_{suffix}'] = stats[f'total_{suffix}'] - stats[f'bad_{suffix}']
        stats[f'bad_rate_{suffix}'] = stats[f'bad_{suffix}'] / stats[f'bad_{suffix}'].sum()
        stats[f'good_rate_{suffix}'] = stats[f'good_{suffix}'] / stats[f'good_{suffix}'].sum()
        stats[f'woe_{suffix}'] = np.log(stats[f'good_rate_{suffix}'] / stats[f'bad_rate_{suffix}'])
        stats.replace({f'woe_{suffix}': {np.inf: 0, -np.inf: 0}}, inplace=True)

        return stats

    # Observed WOE Calculation
    group = df.groupby(feature)
    observed_stats = calculate_woe_stats(group,target, 'obs')
    expected_stats = calculate_woe_stats(group, predicted, 'exp')

    # Combining observed and expected stats
    miv_df = pd.concat([observed_stats, expected_stats], axis=1)

    # Create descriptions
    miv_df.reset_index(inplace=True)
    miv_df.rename(columns={feature: 'bin'}, inplace=True)
    miv_df.insert(0, 'feature', feature)

    # Calculate MIV
    miv_df['delta'] = miv_df['woe_obs'] - miv_df['woe_exp']
    t1 = sum(miv_df['good_obs'] * miv_df['delta']) / sum(miv_df['good_obs'])
    t2 = sum(miv_df['bad_obs'] * miv_df['delta']) / sum(miv_df['bad_obs'])
    miv_df['miv'] = t1 - t2

    # Chi-Square test
    sq_good = miv_df['good_obs'] * np.log(miv_df['good_obs'] / miv_df['good_exp'])
    sq_bad = miv_df['bad_obs'] * np.log(miv_df['bad_obs'] / miv_df['bad_exp'])
    sq_stat = 2 * sum(sq_good + sq_bad)
    miv_df['sq_stat'] = sq_stat
    miv_df['p_val'] = stats.chi2.sf(sq_stat, len(miv_df) - 1)

    return miv_df


def calculate_psi(dataset, date_col, bin_col):
    """
    Calculate the Population Stability Index (PSI) for a given dataset.
    
    :param dataset: DataFrame containing the data
    :param date_col: Name of the column containing dates
    :param bin_col: Name of the column containing bins
    :return: Total PSI value
    """
    # Create a temporary boolean column for the latest date
    latest_date = dataset[date_col].max()
    dataset['is_latest'] = dataset[date_col] == latest_date

    # Create a pivot table with bin counts
    pivot = dataset.pivot_table(index=bin_col, columns='is_latest', aggfunc='size', fill_value=0)

    # Rename columns for clarity
    pivot.columns = ['Not_Latest', 'Latest']

    # Identify rows with 0 in either 'Latest' or 'Not_Latest' columns and add 1 to both columns for those rows
    rows_with_zero = pivot[(pivot['Latest'] == 0) | (pivot['Not_Latest'] == 0)]
    pivot.loc[rows_with_zero.index, ['Latest', 'Not_Latest']] += 1

    # Calculate PSI
    pivot['PSI'] = (pivot['Latest'] / pivot['Latest'].sum() - pivot['Not_Latest'] / pivot['Not_Latest'].sum()) * np.log((pivot['Latest'] / pivot['Latest'].sum()) / (pivot['Not_Latest'] / pivot['Not_Latest'].sum()))
    total_psi = pivot['PSI'].sum()

    return total_psi
    

def gini_coefficient(data, var, target):
    """ Calculate the Gini coefficient for a variable using AUC.
        The function handles both continuous and categorical variables.
    """
    # Check if the variable is categorical
    df = data.dropna(subset=[var, target]).copy()
    
    if df[var].dtype == 'object':
        # Calculate the default rate for each category
        default_rates = df.groupby(var)[target].mean()
        y_scores = df[var].map(default_rates)
    else:
        # Use the variable as is for continuous variables
        y_scores = df[var]

    # Calculate AUC and then Gini coefficient
    auc = roc_auc_score(df[target], y_scores)
    return 2 * auc - 1


def binning_quality_score(iv, p_values, hhi_norm):
    # Score 1: Information value
    c = 0.39573882184806863
    score_1 = iv * np.exp(1/2 * (1 - (iv / c) ** 2)) / c

    # Score 2: statistical significance (pairwise p-values)
    p_values = np.asarray(p_values)
    score_2 = np.prod(1 - p_values)

    # Score 3: homogeneity
    score_3 = 1. - hhi_norm

    return score_1 * score_2 * score_3


def hhi(s, normalized=False):
    """Compute the Herfindahl–Hirschman Index (HHI).

    Parameters
    ----------
    s : array-like
        Fractions (exposure)

    normalized : bool (default=False)
        Whether to compute the normalized HHI.
    """
    s = np.asarray(s)
    h = np.sum(s ** 2)

    if normalized:
        n = len(s)
        if n == 1:
            return 1
        else:
            n1 = 1. / n
            return (h - n1) / (1 - n1)

    return h


def chi_square_test(df, column, target):
    """
    Performs a chi-square test to determine if the missingness in a given column is
    related to a binary target variable.

    Parameters:
    df (DataFrame): The pandas DataFrame containing your data.
    column (str): The name of the column to check for missingness.
    target (str): The name of the binary target variable.

    Returns:
    None: Prints the result of the chi-square test.
    """
    # Create a flag for missing values
    df['is_missing'] = df[column].isna()

    # Create a contingency table
    contingency_table = pd.crosstab(df['is_missing'], df[target])

    # Perform the chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Basic interpretation
    if p < 0.05:
        return "Informative missings", p
    else:
        return "Uninformative missings", p
        

def calculate_hhi(data, column):
    # Calculate the share of each value in the column
    total_sum = data[column].sum()
    shares = data[column] / total_sum

    # Calculate HHI
    hhi = (shares ** 2).sum()  # Multiplying by 10000 for the conventional HHI scale
    return hhi

def calculate_most_important_feature(data, target_column, predictors):
    # Prepare the data
    X = data[predictors]
    y = data[target_column]

    # Fit a logistic regression model
    model = LogisticRegression().fit(X, y)

    # Calculate permutation importance
    try:
        perm_importance = permutation_importance(model, X, y, n_repeats=30, random_state=0)
    
        # Find the feature with the maximum importance
        max_importance_index = np.argmax(perm_importance.importances_mean)
        max_importance_feature = predictors[max_importance_index]
        max_importance_value = perm_importance.importances_mean[max_importance_index]
    
        # Calculate the percentage of the total importance
        total_importance = np.sum(perm_importance.importances_mean)
    
        if total_importance == 0:
            importance_percentage = 100.0
        else:
            importance_percentage = (max_importance_value / total_importance) * 100
    
        return max_importance_feature, importance_percentage

    except:
        np.nan, np.nan

def calculate_RWA(exposure_type, PD, LGD, EAD, add_on=1.06, M=None, S=None):
    """
    Calculate Risk-Weighted Assets (RWA) for different types of exposures.
    
    Parameters:
    - exposure_type (str): Type of regulatory exposure class.
    - PD (float): Probability of Default. Must be between 0 and 1.
    - LGD (float): Loss Given Default. Must be between 0 and 1.
    - EAD (float): Exposure at Default.
    - M (float, optional): Maturity for corporate exposures. Must be between 1 and 3. Default is None.
    - S (float, optional): Sales for SME corporate exposures. Must be between 1 and 50. Default is None.
    
    Returns:
    - float: Risk-Weighted Assets (RWA) based on the given inputs.
    
    Raises:
    - AssertionError: If PD or LGD are not between 0 and 1.
    """
    
    # Validate input ranges
    assert 0 <= PD <= 1, "PD must be between 0 and 1."
    assert 0 <= LGD <= 1, "LGD must be between 0 and 1."
    if M is not None:
        M = np.clip(M, 1, 5)
    if S is not None:
        S = np.clip(S, 5, 50)
    
    # The G function, the inverse of the standard normal cumulative distribution function
    def G(z):
        return norm.ppf(z)

    # Calculate Correlation R based on exposure type
    if exposure_type == "Residential Mortgages":
        R = 0.15
    elif exposure_type == "Qualifying Revolving Retail Exposures":
        R = 0.04
    elif exposure_type == "Other Retail Exposures":
        R = 0.03 * (1 - np.exp(-35 * PD)) / (1 - np.exp(-35)) + 0.16 * (1 - (1 - np.exp(-35 * PD)) / (1 - np.exp(-35)))
    elif exposure_type == "Corporate Exposures":
        R = 0.12 * (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)) + 0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)))
        if S is not None:
            R -= 0.04 * (1 - (S - 5) / 45)
    else:
        return np.nan
    
    # Calculate Capital Requirement K
    if exposure_type == "Corporate Exposures":
        # Maturity adjustment b for corporate exposures
        b = (0.11852 - 0.05478 * np.log(PD)) ** 2
        K = (LGD * norm.cdf(np.sqrt((1 - R)**-1) * G(PD) + np.sqrt(R / (1 - R)) * G(0.999)) - PD * LGD) * (1 - 1.5 * b) ** -1 * (1 + (M - 2.5) * b)
    else:
        K = LGD * norm.cdf(((1 - R)**-0.5) * G(PD) + (R / (1 - R)) ** 0.5 * G(0.999)) - PD * LGD
    
    # Calculate Risk-Weighted Assets RWA
    RWA = K * 12.5 * EAD * add_on
    
    return RWA