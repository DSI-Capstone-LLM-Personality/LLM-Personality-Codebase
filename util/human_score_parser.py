import json
import pandas as pd
import zipfile
import argparse
from collections import defaultdict
import math
import sys
import numpy as np



def get_stats(humanAns, questions, version):
    traits_distribution = defaultdict(int)
    traits_mean = defaultdict(int)
    traits_std = defaultdict(int)
    
    human_scores = humanAns.iloc[:,11:]
    human_scores[~(human_scores == 0).all(axis=1)]
    human_scores = human_scores.astype(int)
    human_scores_no_zero = human_scores.replace(0, np.nan)

    
    indices = []
    trait = []

    for i in range(version):
        if version == 120:
            indices.append(int(questions["Short#"][i]))
            trait.append(questions["Key"][i])
        elif version == 300:
            indices.append(int(questions["Full#"][i]))
            trait.append(questions["Key"][i])
        else:
            print("Wrong data path for human answers")
            
   #pandas calculates mean/var properly with na, but not 0. 

    for i in range(version):
        idx = indices[i]
        t = trait[i][:1]
        traits_distribution[t] += 1
        s = human_scores_no_zero.iloc[:,idx-1:idx].mean(skipna=True, numeric_only=False,)
        traits_mean[t] += s[0]
        var = human_scores_no_zero.iloc[:,idx-1:idx].var(skipna=True, numeric_only=False,)
        traits_std[t] += var[0]
    
    
#     print("Traits Distribution: \n", traits_distribution)
    
    total = traits_distribution["N"]
    traits_mean = {key: value / total for key, value in traits_mean.items()}
#     print("\n\n Traits Mean: \n", traits_mean)
    
    traits_std = {key: math.sqrt(value)/math.sqrt(total) for key, value in traits_std.items()}
#     print("\n\n Traits Standard Deviation: \n", traits_std)
    
    
    return traits_distribution, traits_mean, traits_std


    
def process_results(dset, logfile, traits_mean,traits_std ):
    original_stdout = sys.stdout
    
    with open(logfile, 'w') as f:
        sys.stdout = f
        print(f"DATASET: {dset}")
        print("\n\nOCEAN SCORES STATS\n")
      
        OCEAN = ['O','C','E', 'A','N']
        
        for i in range(len(OCEAN)):
            m = traits_mean.get(OCEAN[i])
            std = traits_std.get(OCEAN[i])
            print(f"{OCEAN[i]} | MEAN: {np.round(m, 5):<8} | STD: {np.round(std, 5)}")
       
        f.close()
        sys.stdout = original_stdout    
    
    
    
    
def main():
    """
    Main file to run from the command line.
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--humanAns",
                        default="IPIP120.csv.zip",
                        help="filename for human answers to the personality test")
    parser.add_argument("--questions",
                        default="IPIP-NEO-ItemKey.xls",
                        help="filename for labels associated with human answers")


    args = parser.parse_args()
    humanAns = pd.read_csv(args.humanAns)
    questions = pd.read_excel(args.questions)
    
    version = args.humanAns.split(".")[0]
    datafile = version[:7]
    version = int(version[4:])
    
    
    
    print("-------------Info------------")
    print("Number of questions:", version)
    print("Dataset for human answers:", args.humanAns)
    print("Dataset for ItemKey:", args.questions)
    print("-------------Info------------")
    print("\n\n")
    
    traits_distribution, traits_mean, traits_std = get_stats(humanAns, questions, version)
    
    # LOG_FILE_NAME = f"[{datafile}]_[ans-stats].txt"
    LOG_FILE_NAME = f"Dataset/Human Data/results/[{datafile}]_[ans-stats].txt"
    process_results(args.humanAns, LOG_FILE_NAME, traits_mean, traits_std)



if __name__ == "__main__":
    main()
    
    