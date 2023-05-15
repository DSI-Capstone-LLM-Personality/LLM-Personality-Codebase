import os
import torch
import argparse
from MPI.mpi import *
from collections import defaultdict



if __name__ == "__main__":
    # Parse Input Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--chpk-dir', help='path to checkpoints')
    args = parser.parse_args()

    # Test
    if not args.chpk_dir:
        args.chpk_dir = 'checkpoint/mpis/Constraint/order-symmetry/opt-30b/non-index/desc'

    checkpoints = os.listdir(args.chpk_dir)

    mpis = defaultdict(lambda: defaultdict(list))
    for chkp in checkpoints:
        mpi = torch.load(os.path.join(args.chpk_dir, chkp))
        template_name = chkp.split('_')[-4]
        order = chkp.split('_')[-2][1:-1]
        start = int(chkp.split('#')[1])
        end = int(chkp.split('#')[2])-1
        chkp = chkp.split('.')[0]
        mpis[template_name][order].append(dict(start=start,end=end,mpi=mpi,filename=chkp))
    
    for template, order_dct in mpis.items():
        for order, mpi_lst in order_dct.items():
            mpi_lst = sorted(mpi_lst,key= lambda k: k['start'])

            reduced_mpi = mpi_lst[0]['mpi']

            for k in OCEAN:
                reduced_mpi.OCEAN[k].extend([])
            for idx in range(1,len(mpi_lst)):
                mpi = mpi_lst[idx]['mpi']
                reduced_mpi.plus_minus = np.hstack([reduced_mpi.plus_minus,mpi.plus_minus])
                reduced_mpi.preds_key = np.hstack([reduced_mpi.preds_key,mpi.preds_key])
                reduced_mpi.label = np.hstack([reduced_mpi.label,mpi.label])
                reduced_mpi.key = torch.cat((reduced_mpi.key,mpi.key),0)
                reduced_mpi.likelihood.extend(mpi.likelihood)
                reduced_mpi.preds.extend(mpi.preds)
                reduced_mpi.probs.extend(mpi.probs)
                reduced_mpi.questions = np.hstack([reduced_mpi.questions,mpi.questions])
                reduced_mpi.scores.extend(mpi.scores)
                reduced_mpi.text = np.hstack([reduced_mpi.text,mpi.text])
                reduced_mpi.token_of_interest.extend(mpi.token_of_interest)
                reduced_mpi.mpi_df = pd.concat([reduced_mpi.mpi_df,mpi.mpi_df])
                
                # for k in OCEAN:
                #     mpi.OCEAN[k].extend([])

                for key in reduced_mpi.OCEAN.keys():
                    reduced_mpi.OCEAN[key].extend(mpi.OCEAN[key])

            log_dir = args.chpk_dir.replace('mpis','log')
            filename = '_'.join(mpi_lst[0]['filename'].split('_')[:-3]+[f'[{order}]']+mpi_lst[0]['filename'].split('_')[-1:])

            reduced_mpi.write_statistic(os.path.join(log_dir,filename + '.txt'))
            reduced_mpi.save_checkpoint(os.path.join(args.chpk_dir, filename + '.pt'))