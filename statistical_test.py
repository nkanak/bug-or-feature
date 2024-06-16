# Standard Library Imports
import json
import sys

# Scientific Computing Imports
from scipy.stats import wilcoxon, ttest_rel, ttest_ind

# Define the folder and result files
folder = 'results/'
result_files = ['results_jira_0.json', 'results_jira_1.json', 'results_jira_2.json']

def statistical_tests(test='wilcoxon'):
    """
    Perform statistical significance tests on the results of different models.

    Parameters:
    test (str): The type of test to perform. Options are 'wilcoxon', 'ttest_rel', 'ttest_ind'.

    Raises:
    ValueError: If the test parameter is not one of the allowed values.
    """
    # Validate the test parameter
    if test not in ['wilcoxon', 'ttest_rel', 'ttest_ind']:
        raise ValueError('The test should be either wilcoxon, ttest_rel, ttest_ind')

    # Map the test parameter to the corresponding function
    test_func = eval(test)
    
    # Iterate through each result file
    for file in result_files:
        print(f'Statistical Significance Test for dataset {file}\n-----------------------------------------------------------\n')
        
        # Load the results from the JSON file
        with open(folder + file, 'r') as f:
            results = json.load(f)
        
        # Extract GAT results and remove from dictionary for comparison
        our_approach_results = results['GAT']
        del results['GAT']

        # Iterate through each model in the results
        for model in results.keys():
            print(f'Statistical Test Between GAT and {model}\n--------------------------------------------')

            # Iterate through each metric in the GAT results
            for metric in our_approach_results.keys():
                # Perform the appropriate statistical test
                if test_func == wilcoxon:
                    if model in ['LR', 'KNN']:
                        stat, p_value = wilcoxon(our_approach_results[metric], results[model][metric] * 10)
                    else:
                        stat, p_value = wilcoxon(our_approach_results[metric], results[model][metric])
                elif test_func == ttest_rel:
                    if model in ['LR', 'KNN']:
                        stat, p_value = ttest_rel(our_approach_results[metric], results[model][metric] * 10)
                    else:
                        stat, p_value = ttest_rel(our_approach_results[metric], results[model][metric])
                else:
                    if model in ['LR', 'KNN']:
                        stat, p_value = ttest_ind(our_approach_results[metric], results[model][metric] * 10)
                    else:
                        stat, p_value = ttest_ind(our_approach_results[metric], results[model][metric])
                
                # Print the results of the statistical test
                print(f'{metric}|\tStatistic: {stat}\tp_value: {p_value}')
            
            print('\n')

        print('\n\n\n')

if __name__ == '__main__':
    """
    Entry point of the script. If no command-line arguments are provided, defaults to using 'wilcoxon' test.
    """
    if len(sys.argv) <= 1:
        print('Running default statistical test (Wilcoxon)')
        statistical_tests(test='wilcoxon')
    else:
        test = sys.argv[1]
        print(f'{test.capitalize()} Statistical Significance Test\n---------------------------------------------------------\n\n')
        statistical_tests(test=test)
