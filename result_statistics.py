import numpy as np
import json
import sys

def print_evaluation_results(results):
    """
    Print the average accuracy, precision, and recall with their standard deviations.

    Parameters:
    results (list of tuples): A list where each tuple contains accuracy, precision, and recall values.
    """
    print('Avg accuracy | Avg precision | Avg recall')
    
    # Calculate the average and standard deviation for accuracy, precision, and recall
    avg_accuracy, avg_precision, avg_recall = np.mean(results, axis=0)
    std_accuracy, std_precision, std_recall = np.std(results, axis=0)
    
    # Print the results with standard deviations
    print(f'{avg_accuracy*100:.2f}+-{std_accuracy*100:.2f}, {avg_precision*100:.2f}+-{std_precision*100:.2f}, {avg_recall*100:.2f}+-{std_recall*100:.2f}')

def main(file):
    """
    Main function to load results from a JSON file and print evaluation results for each model.

    Parameters:
    file (str): The name of the JSON file containing the results.
    """
    folder = 'results/'  # Define the folder where the results are stored
    
    # Open and load the JSON file
    with open(folder + file, 'r') as f:
        results = json.load(f)
    
    # Print the dataset name (extracted from the filename)
    print(f'Dataset: {file.split("_", maxsplit=1)[1]}')
    
    # Loop through each model in the results
    for model in results.keys():
        print(f'Model {model}')
        
        # Extract accuracy, precision, and recall for the model
        accuracy = results[model]['Accuracy']
        precision = results[model]['Precision']
        recall = results[model]['Recall']
        
        # Combine accuracy, precision, and recall into a list of tuples
        model_results = list(zip(accuracy, precision, recall))
        
        # Print evaluation results for the model
        print_evaluation_results(model_results)
        print('-----------------------------------------------------------')
    
    print('----------------------------------')

if __name__ == '__main__':
    """
    Entry point of the script. If no command-line arguments are provided, default to 'results_jira_1.json'.
    """
    if len(sys.argv) <= 1:
        # Default file
        main('results_jira_0.json')  
    else:
        # File provided as a command-line argument
        main(sys.argv[1])  