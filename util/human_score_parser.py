import json
import pandas as pd
import zipfile
import argparse
from collections import defaultdict
import math



def get_stats(humanAns, questions, version):
    traits_distribution = defaultdict(int)
    traits_mean = defaultdict(int)
    traits_std = defaultdict(int)
    
    human_scores = humanAns.iloc[:,11:]
    
    indices = []
    trait = []
#     num = 300

    for i in range(version):
        if version == 120:
            indices.append(int(questions["Short#"][i]))
            trait.append(questions["Key"][i])
        elif version == 300:
            indices.append(int(questions["Full#"][i]))
            trait.append(questions["Key"][i])
        else:
            print("Wrong data path for human answers")
    
    for i in range(version):
        idx = indices[i]
        t = trait[i][:1]
        traits_distribution[t] += 1
        s = human_scores.iloc[:,idx-1:idx].mean()
        traits_mean[t] += s[0]
        var = human_scores.iloc[:,idx-1:idx].var()
        traits_std[t] += var[0]
    
    
    
    print("Traits Distribution: \n", traits_distribution)
    
    total = traits_distribution["N"]
    traits_mean = {key: value / total for key, value in traits_mean.items()}
    print("\n\n Traits Mean: \n", traits_mean)
    
    traits_std = {key: math.sqrt(value/math.sqrt(total)) for key, value in traits_std.items()}
    print("\n\n Traits Standard Deviation: \n", traits_std)

    
    
    
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
    version = int(version[4:])
    
    
    print("-------------Info------------")
    print("Number of questions:", version)
    print("Dataset for human answers:", args.humanAns)
    print("Dataset for ItemKey:", args.questions)
    print("-------------Info------------")
    print("\n\n")
    
    get_stats(humanAns, questions, version)



if __name__ == "__main__":
    main()
    
    