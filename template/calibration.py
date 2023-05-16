import os
import sys
import yaml
import json
import argparse

# Required for project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MPI.mpi import *
from util.utils import *
from template.templates import *
from model.language_model import *


def calculate_scaling_vectors(args):

    ## Config Load ----------------------------------
    assert args.config is not None, 'Please specify the config .yaml file to proceed.'
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    set_seed(args.seed)

    # Regime
    regime, category = config['experiment'].values()
    assert regime in ['Constraint']
    assert category in ['calibration']

    # Template
    tmp = config['template']
    index = tmp['index']
    description = tmp['description']
    if index is not None:
        index = {k: INDEX[v] for k, v in index.items()}
    assert tmp['desc'] is not None
    desc = {k: DESC[v] for k, v in tmp['desc'].items()}
    ans_type = tmp['ans_type']

    # Order Shuffle
    if config['shuffle'] is not None:
        shuffle_both = config['shuffle']['shuffle_both']
    else: shuffle_both = False
    
    # Model and tokenizer
    family, version, access_method = args.family, args.version, ACCESS[regime][args.family]
    assert access_method == 'hf', f'Access method: {access_method} is not implemeted'
    tokenizer = TOKENIZER[family].from_pretrained(version)
    model = MODEL[regime][family].from_pretrained(version, pad_token_id=tokenizer.eos_token_id).to(DEVICE)
    ll_type = config['algorithm']['ll_type']
    prober = LMPROB(family, model, tokenizer, ll_type, False)
    assert ll_type == 'ans_inv_perp', f'll_type: {ll_type} is not implemeted'
    if family in ['OPT', 'GPTNEO', 'GPTNEOX']:
        version = version.split('/')[1]

    # Path
    path = config['path']
    tmp_dir = path['tmp_dir']
    ckpt_dir = path['ckpt_dir']
    log_dir = path['log_dir']
    log_dir = os.path.join(log_dir, regime, category, version)
    ckpt_dir = os.path.join(ckpt_dir, regime, category, version)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)


    ## Calibration Exp ----------------------------------

    print('\nCalculating Scaling Vectors...\n\n')

    # Get templates
    templates = [f.split('.')[0] for f in os.listdir(tmp_dir) if f.endswith('.txt')]
    lc_templates = ['[lc]'+ tmp[4:] for tmp in templates]
    templates = templates + lc_templates

    # Get orders
    all_orders = list(ORDERS.keys())
    
    # Set logger
    filename = f'[{family}]_[{version}]_[{description}]_[{ans_type}]'
    logger = open(os.path.join(log_dir,f'{filename}.log'),'w')

    # Banner
    logger.write('\nCalibration Experiment\n')
    logger.write(f'{"-"*50}\n')
    logger.write(f'Family    : {family}\n')
    logger.write(f'Version   : {version}\n')
    logger.write(f'Device    : {DEVICE.type.upper()}\n')
    if 'cuda' in DEVICE.type:
        logger.write(f'GPU       : {torch.cuda.get_device_name(0)}\n')
        logger.write(f'Memory    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n\n')
    
    # Progress bars
    choice_pbar     = tqdm(desc='Choice   ', colour='#EF5350', total=len(desc['+']))
    template_pbar   = tqdm(desc='Template ', colour='#E0E0E0', total=len(templates))
    order_pbar      = tqdm(desc='Order    ', colour='#42A5F5', total=len(all_orders))
    history = dict(family=family, version=version, index_type=description, answer_type=ans_type, orders=defaultdict(dict))

    for ord_idx, ord in enumerate(all_orders,1):
        # order
        order = ORDERS[ord]

        history['orders'][ord] = defaultdict(dict)
        logger.write(f'{"_"*100}\n\n')

        template_pbar.reset()
        for tmp_idx, template in enumerate(templates,1):
            history['orders'][ord][template]=defaultdict(dict)
            logger.write(f'Order {ord_idx}  : {ord} {order}\n')
            logger.write(f'Template {tmp_idx}  : {template}\n')
           
            # lower case
            is_lower_case = True if template[1:3]=='lc' else False
            if is_lower_case: sent_template = '[og]'+ template[4:]
            else: sent_template = template
        
            # options
            option_formatter = MPIOptionFormatter(index, desc, is_lower_case)
            option = option_formatter(order, shuffle_both)

            # template
            prompt_template = get_template(sent_template)
            prompt_template = prompt_template.split('"')[0] + '"{item}"'+ prompt_template.split('"')[2]
            
            # question
            pformatter = MPIQuestionFormatter(prompt_template, option)
            prompt = pformatter(' ','+')
            
            logger.write(f'\nPrompt :  \n{prompt}\n\n')
            logger.write(f'{"-"*40}\n\n')
            
            # choice list
            choice_lst = MPI_options_to_answers(index, desc, option, ans_type, is_lower_case, order)

            ilp_list = []
            choice_dict = defaultdict(dict)
            choice_pbar.reset()
            for choice in choice_lst['+']:
                prob, ll, toi = prober(prompt, choice)

                ilp_list.append(ll.item())

                logger.write(f'Choice :  {choice}\n')
                logger.write(f'Tokens :  {tokenizer.convert_ids_to_tokens(toi)}\n')
                logger.write(f'IDs    :  {[t.item() for t in toi]}\n')
                logger.write(f'Prob   :  {prob.tolist()}\n') 
                logger.write(f'ILP    :  {ilp_list[-1]}\n\n')

                c_dict = defaultdict(dict)
                c_dict['tokens'] = tokenizer.convert_ids_to_tokens(toi)
                c_dict['ids'] = [t.item() for t in toi]
                c_dict['prob'] = prob.tolist()
                c_dict['ilp'] = ilp_list[-1]
                choice_dict[choice] = c_dict

                choice_pbar.update(1)
            choice_pbar.refresh()
            
            history['orders'][ord][template]['choices']=choice_dict

            exps = torch.exp(torch.tensor(ilp_list))
            min_value, max_value = torch.min(exps), torch.max(exps)
            normalized = (exps - min_value) / (max_value - min_value)
            probabilities = torch.nn.functional.softmax(normalized,0)
            scaling_vector = 1/probabilities
            history['orders'][ord][template]['scaling_vector'] = scaling_vector.tolist()

            logger.write(f"\nScaling Vector: {history['orders'][ord][template]['scaling_vector']}\n\n")
            logger.write(f'\n{"-"*80}\n\n')

            template_pbar.update(1)
            
        template_pbar.refresh()
        order_pbar.update(1)
    
    choice_pbar.close()
    template_pbar.close()
    order_pbar.close()

    # Save history
    with open(os.path.join(ckpt_dir, f'{filename}.json'),'w') as file:
        json.dump(history, file, indent=4)

    logger.write(f'\n\nHistory saved. {filename}.json')
    logger.close()

    print('\n\nProcess finished.\n')


def rescore(args, mpi,chkp, ques_pbar, chkp_pbar, sharded=False):
    ques_pbar.total = len(mpi.scores)
    ques_pbar.refresh()

    dir_splits = args.chpk_dir.split('/')
    family = mpi.model_desc['family']
    version = mpi.model_desc['version']
    if family in ['OPT', 'GPTNEO', 'GPTNEOX']:
        mpi.model_desc['version']=mpi.model_desc['version'].split('/')[0]+'/'+mpi.model_desc['version'].split('/')[1].lower()
        version = mpi.model_desc['version'].split('/')[1]
    description = dir_splits[-2]
    ans_type = dir_splits[-1]
    regime = dir_splits[2]
    ckpt_dir = dir_splits[0]+'/'+dir_splits[1]
    template_name = chkp.split('_')[-3]
    order = chkp.split('_')[-2][1:-1]
    
    hs_dir = os.path.join(ckpt_dir, regime, 'calibration', version)
    hs_filename = f'[{family}]_[{version}]_[{description}]_[{ans_type}].json'
    with open(os.path.join(hs_dir, f'{hs_filename}'),'r') as file:
        scaling_vector = torch.tensor(json.load(file)['orders'][order][template_name]['scaling_vector'])

    mpi.preds_key, mpi.preds = [], []
    mpi.OCEAN = defaultdict(list)
    mpi.scores = []
    ques_pbar.reset()
    for idx, ilp in enumerate (mpi.likelihood):         
        exps = torch.exp(torch.tensor(ilp.tolist()))
        min_value, max_value = torch.min(exps), torch.max(exps)
        normalized = (exps - min_value) / (max_value - min_value)
        probabilities = torch.nn.functional.softmax(normalized,0)

        scaled_probabilities = torch.mul(scaling_vector, probabilities)

        key = mpi.plus_minus[idx]
        pred = torch.argmax(scaled_probabilities,0).item()
        mpi.preds.append(pred)
        mpi.preds_key.append(mpi.mpi_choice_lst[key][pred])
        
        score = MPI_SCORE[key][pred]
        mpi.scores.append(score)
        ques_pbar.update(1)
    ques_pbar.refresh()

    mpi.preds_key = np.array(mpi.preds_key)
    mpi.calculate_score()

    log_dir = args.chpk_dir.replace('mpis','log')
    mpi.write_statistic(os.path.join(log_dir, chkp[:-3] + '_[calibrated].txt'))
    torch.save(mpi, os.path.join(args.chpk_dir, chkp[:-3] + '_[calibrated].pt'))

    chkp_pbar.update(1)

def calibrate_scores(args):

    print('\nScore Calibration in progress...\n\n')

    checkpoints = [ch for ch in os.listdir(args.chpk_dir) if 'calibrated' not in ch]
    sharded = any('#' in filename for filename in checkpoints)

    ques_pbar = tqdm(desc='Question   ', colour='#42A5F5')
    chkp_pbar = tqdm(desc='Checkpoint ', colour='#E0E0E0', total=len(checkpoints))
    
    if sharded:
        mpis = defaultdict(lambda: defaultdict(list))
        for chkp in checkpoints:
            mpi = torch.load(os.path.join(args.chpk_dir, chkp), map_location=DEVICE)
            template_name = chkp.split('_')[-4]
            order = chkp.split('_')[-2][1:-1]
            start = int(chkp.split('#')[1])
            end = int(chkp.split('#')[2])-1
            chkp = chkp.split('.')[0]
            mpis[template_name][order].append(dict(start=start,end=end,mpi=mpi,filename=chkp))



        for template, order_dct in mpis.items():
            chkp_pbar.total = len(mpis[template])
            chkp_pbar.refresh()
            
            for order, mpi_lst in order_dct.items():
                mpi_lst = sorted(mpi_lst,key= lambda k: k['start'])
                chkp = mpis[template][order][0]['filename']
                chkp = '_'.join(chkp.split('_')[:-3]+chkp.split('_')[-2:])+'.pt'

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
                    
                    for key in reduced_mpi.OCEAN.keys():
                        reduced_mpi.OCEAN[key].extend(mpi.OCEAN[key])

                mpi = reduced_mpi

                rescore(args, mpi, chkp, ques_pbar, chkp_pbar, True)

            chkp_pbar.close()
            ques_pbar.close()  

    else:
        for chkp in checkpoints:
            mpi = torch.load(os.path.join(args.chpk_dir, chkp), map_location=DEVICE)

            rescore(args, mpi, chkp, ques_pbar, chkp_pbar)

        chkp_pbar.close()
        ques_pbar.close()  
    
    print('\n\nProcess finished.\n')



if __name__=='__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file')
    parser.add_argument('--mode', help='SV: calculate scaling vectors, CL: calibrate scores', type=str)
    parser.add_argument('--seed', help='python seed', type=int, default=2023)

    # SV mode args
    parser.add_argument('--family', help='model family', type=str)
    parser.add_argument('--version', help='model checkpoint version', type=str)

    # CL mode args
    parser.add_argument('--chpk-dir', help='path to checkpoints')

    args = parser.parse_args()

    # Test
    if not args.config:
        # args.mode = 'SV'
        # args.config = 'config/Constraint/calibration/non-index.yaml'
        # args.family = 'GPTNEOX'
        # args.version = 'EleutherAI/gpt-neox-20b'

        args.mode = 'CL'
        args.chpk_dir = 'checkpoint/mpis/Constraint/order-symmetry/opt-30b/index/index'

    if args.mode == 'SV':
        calculate_scaling_vectors(args)
    elif args.mode =='CL':
        calibrate_scores(args)
