"""
Created on May 22, 2016

This should be a simple minimalist run file. It's only responsibility should be to parse the arguments (which agent, user simulator to use) and launch a dialog simulation.

Rule-agent: python run.py --agt 6 --usr 1 --max_turn 40 --episodes 150 --movie_kb_path .\deep_dialog\data\movie_kb.1k.p --run_mode 2

kb:
movie_kb.1k.p: 94% success rate
movie_kb.v2.p: 36% success rate

user goal files:
first turn: user_goals_first_turn_template.v2.p
all turns: user_goals_all_turns_template.p
user_goals_first_turn_template.part.movie.v1.p: a subset of user goal. [Please use this one, the upper bound success rate on movie_kb.1k.json is 0.9765.]

Commands:
Rule: python run.py --agt 5 --usr 1 --max_turn 40 --episodes 150 --kb_path .\deep_dialog\data\movie_kb.1k.p --goal_file_path .\deep_dialog\data\\user_goals_first_turn_template.part.movie.v1.p --intent_err_prob 0.00 --slot_err_prob 0.00 --episodes 500 --act_level 1 --run_mode 1

Training:
RL: python run.py --agt 9 --usr 1 --max_turn 40 --kb_path .\deep_dialog\data\movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir .\deep_dialog\checkpoints\rl_agent\ --run_mode 3 --act_level 0 --slot_err_prob 0.05 --intent_err_prob 0.00 --batch_size 16 --goal_file_path .\deep_dialog\data\\user_goals_first_turn_template.part.movie.v1.p --warm_start 1 --warm_start_epochs 120

Predict:
RL: python run.py --agt 9 --usr 1 --max_turn 40 --kb_path .\deep_dialog\data\movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 300 --simulation_epoch_size 100 --write_model_dir .\deep_dialog\checkpoints\rl_agent\ --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path .\deep_dialog\data\\user_goals_first_turn_template.part.movie.v1.p --episodes 200 --trained_model_path .\deep_dialog\checkpoints\rl_agent\agt_9_22_30_0.37000.p --run_mode 3

@author: xiul, t-zalipt
"""

import argparse, json, copy, os, csv
import pickle as pickle

import torch
from collections import deque

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, AgentDQN, RequestInformSlotAgent
from deep_dialog.usersims import RuleSimulator, RuleRestaurantSimulator, RuleTaxiSimulator

from deep_dialog import dialog_config
from deep_dialog.dialog_config import *

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""

""" load action """
def load_actions(sys_req_slots, sys_inf_slots):
    dialog_config.feasible_actions = [
        {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
    ]

    for slot in sys_inf_slots:
        dialog_config.feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

    for slot in sys_req_slots:
        dialog_config.feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dict_path', dest='dict_path', type=str, default='./deep_dialog/data_restaurant/slot_dict.v2.p', help='path to the .json dictionary file')
    parser.add_argument('--kb_path', dest='kb_path', type=str, default='./deep_dialog/data_restaurant/restaurant.kb.1k.v1.p', help='path to the movie kb .json file')
    parser.add_argument('--act_set', dest='act_set', type=str, default='./deep_dialog/data_restaurant/dia_acts.txt', help='path to dia act set; none for loading from labeled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default='./deep_dialog/data_restaurant/restaurant_slots.txt', help='path to slot set; none for loading from labeled file')
    parser.add_argument('--goal_file_path', dest='goal_file_path', type=str, default='./deep_dialog/data_restaurant/user_goals_first.v1.p', help='a list of user goals')
    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str, default='./deep_dialog/data_restaurant/sim_dia_act_nl_pairs.v2.json', help='path to the pre-defined dia_act&NL pairs')

    parser.add_argument('--max_turn', dest='max_turn', default=20, type=int, help='maximum length of each dialog (default=20, 0=no maximum length)')
    parser.add_argument('--episodes', dest='episodes', default=1, type=int, help='Total number of episodes to run (default=1)')
    parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.05, type=float, help='the slot err probability')
    parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int, help='slot_err_mode: 0 for slot_val only; 1 for three errs')
    parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.05, type=float, help='the intent err probability')
    
    parser.add_argument('--agt', dest='agt', default=0, type=int, help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
    parser.add_argument('--usr', dest='usr', default=0, type=int, help='Select a user simulator. 0 is a Frozen user simulator.')
    
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0, help='Epsilon to determine stochasticity of epsilon-greedy agent policies')
    
    # load NLG & NLU model
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str, default='./deep_dialog/models/nlg/restaurant/lstm_tanh_[1532068150.19]_98_99_294_0.983.p', help='path to model file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str, default='./deep_dialog/models/nlu/restaurant/lstm_[1532107808.26]_68_74_20_0.997.p', help='path to the NLU model file')
    
    parser.add_argument('--act_level', dest='act_level', type=int, default=0, help='0 for dia_act level; 1 for NL level')
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=0, help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
    parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0, help='0 for no auto_suggest; 1 for auto_suggest')
    parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0, help='run_mode: 0 for NL; 1 for dia_act')
    
    # RL agent parameters
    parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size', type=int, default=1000, help='the size for experience replay')
    parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=80, help='the hidden size for DQN')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
    parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
    parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=50, help='the size of validation set')
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1, help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100, help='the number of epochs for warm start')
    
    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None, help='the path for trained model')
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str, default='./deep_dialog/checkpoints/', help='write model to disk') 
    parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10, help='number of epochs for saving model')
     
    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.3, help='the threshold for success rate')
    
    parser.add_argument('--split_fold', dest='split_fold', default=5, type=int, help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase', default='all', type=str, help='train/test/all; default is all')
    parser.add_argument('--dueling_dqn', type=int, default=0)
    parser.add_argument('--double_dqn', type=int, default=0)
    parser.add_argument('--icm', type=int, default=0)
    parser.add_argument('--per', type=int, default=0)
    parser.add_argument('--noisy', type=int, default=0)
    parser.add_argument('--distributional', type=int, default=0)
    
    args = parser.parse_args()
    params = vars(args)

    print('Dialog Parameters: ')
    print(json.dumps(params, indent=2))


max_turn = params['max_turn']
num_episodes = params['episodes']

agt = params['agt']
usr = params['usr']

dict_path = params['dict_path']
goal_file_path = params['goal_file_path']

# load the user goals from .p file
all_goal_set = pickle.load(open(goal_file_path, 'rb'), encoding='utf-8')

# split goal set
split_fold = params.get('split_fold', 5)
goal_set = {'train':[], 'valid':[], 'test':[], 'all':[]}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1: goal_set['test'].append(u_goal)
    else: goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)
# end split goal set

kb_path = params['kb_path']
kb = pickle.load(open(kb_path, 'rb'), encoding='utf-8')

act_set = text_to_dict(params['act_set'])
slot_set = text_to_dict(params['slot_set'])

################################################################################
# a movie dictionary for user simulator - slot:possible values
################################################################################
movie_dictionary = pickle.load(open(dict_path, 'rb'), encoding='utf-8')

dialog_config.run_mode = params['run_mode']
dialog_config.auto_suggest = params['auto_suggest']

################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']

agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']
agent_params['dueling_dqn'] = params['dueling_dqn']
agent_params['double_dqn'] = params['double_dqn']
agent_params['icm'] = params['icm']
agent_params['per'] = params['per']
agent_params['noisy'] = params['noisy']
agent_params['distributional'] = params['distributional']

if agt == 0:
    agent = AgentCmd(kb, act_set, slot_set, agent_params)
elif agt == 1:
    agent = InformAgent(kb, act_set, slot_set, agent_params)
elif agt == 2:
    agent = RequestAllAgent(kb, act_set, slot_set, agent_params)
elif agt == 3:
    agent = RandomAgent(kb, act_set, slot_set, agent_params)
elif agt == 4:
    #agent = EchoAgent(kb, act_set, slot_set, agent_params)
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, movie_request_slots, movie_inform_slots)
elif agt == 5: # movie request rule agent
    agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, movie_request_slots)
elif agt == 6: # restaurant request rule agent
    agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, restaurant_request_slots)
elif agt == 7: # taxi request agent
    agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, taxi_request_slots)
elif agt == 8: # taxi request-inform rule agent
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, taxi_request_slots, taxi_inform_slots)
elif agt == 9: # DQN agent for movie domain
    agent = AgentDQN(kb, act_set, slot_set, agent_params)
    agent.initialize_config(movie_request_slots, movie_inform_slots)
elif agt == 10: # restaurant request-inform rule agent
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, restaurant_request_slots, restaurant_inform_slots)
elif agt == 11: # taxi request-inform-cost rule agent
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, taxi_request_slots, taxi_inform_cost_slots)
elif agt == 12: # DQN agent for restaurant domain
    load_actions(dialog_config.restaurant_sys_request_slots, dialog_config.restaurant_sys_inform_slots)
    agent = AgentDQN(kb, act_set, slot_set, agent_params)
    agent.initialize_config(restaurant_request_slots, restaurant_inform_slots)
elif agt == 13: # DQN agent for taxi domain
    load_actions(dialog_config.taxi_sys_request_slots, dialog_config.taxi_sys_inform_slots)
    agent = AgentDQN(kb, act_set, slot_set, agent_params)
    agent.initialize_config(taxi_request_slots, taxi_inform_slots)
    
################################################################################
#    Add your agent here
################################################################################
else:
    pass

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']

if usr == 0:# real user
    user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 1: # movie simulator
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 2: # restaurant simulator
    user_sim = RuleRestaurantSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 3: # taxi simulator
    user_sim = RuleTaxiSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)


################################################################################
#    Add your user simulator here
################################################################################
else:
    pass


################################################################################
# load trained NLG model
################################################################################
nlg_model_path = params['nlg_model_path']
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs) # load nlg templates

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)

################################################################################
# load trained NLU model
################################################################################
nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)

################################################################################
# Dialog Manager
################################################################################
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, kb)
    
################################################################################
#   Run num_episodes Conversation Simulations
################################################################################
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size'] # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']


""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward':float('-inf'), 'ave_turns': float('inf'), 'epoch':0}
best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}
performance_records['ave_intrinsic_reward'] = {}


""" Save model """
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.p' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    if agt == 9:
        checkpoint['state_dict'] = {k: v.cpu() for k, v in list(agent.dqn.state_dict().items())}
    if (agt == 12 or agt == 13): checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    checkpoint['params'] = params
    checkpoint['agent_params'] = agent_params
    try:
        torch.save(checkpoint, open(filepath, "wb+"))
        print('saved model in %s' % (filepath, ))
    except Exception as e:
        print('Error: Writing model fails: %s' % (filepath, ))
        print(e)

""" save performance numbers """
def save_performance_records(path, agt, records):
    filename = 'agt_%s_performance_records.json' % (agt)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "wb"))
        print('saved model in %s' % (filepath, ))
    except Exception as e:
        print('Error: Writing model fails: %s' % (filepath, ))
        print(e)

""" Run N simulation Dialogues """
def simulation_epoch(simulation_epoch_size, train=False):
    successes = 0
    cumulative_reward = 0
    intrinsic_reward = 0
    cumulative_turns = 0
    loss = 0
    update_count = 0
    step = 0
    
    res = {}
    for episode in range(simulation_epoch_size):
        dialog_manager.initialize_episode()
        episode_over = False
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            #cumulative_intrinsic_reward += dialog_manager.instrinsic_reward
            cumulative_reward += reward
            step += 1
            if episode_over:
                if reward > 0: 
                    successes += 1
                    #print ("simulation episode %s: Success" % (episode))
                #else: print ("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
            if train and step % 1 == 0:
                err, i_r = agent.train(batch_size, 1)
                loss += err
                intrinsic_reward += i_r
                update_count += 1
    if train:
        print(("cur bellman err %.4f, experience replay pool %s" % (loss/(update_count+1e-10), len(agent.experience_replay_pool))))
    res['success_rate'] = float(successes)/simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward)/simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns)/simulation_epoch_size
    res['ave_intrinsic_reward'] = float(intrinsic_reward)/simulation_epoch_size
    print(("simulation success rate %s, ave reward %s, ave turns %s, i_r %s" % (res['success_rate'], res['ave_reward'], res['ave_turns'], res['ave_intrinsic_reward'])))
    return res

""" Warm_Start Simulation (by Rule Policy) """
def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    res = {}
    warm_start_run_epochs = 0
    for episode in range(warm_start_epochs):
        dialog_manager.initialize_episode()
        episode_over = False
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            if episode_over:
                if reward > 0: 
                    successes += 1
                #    print ("warm_start simulation episode %s: Success" % (episode))
                #else: print ("warm_start simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
        
        warm_start_run_epochs += 1
        
        if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
            break

    agent.warm_start = 2
    res['success_rate'] = float(successes)/warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward)/warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns)/warm_start_run_epochs
    print(("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (episode+1, res['success_rate'], res['ave_reward'], res['ave_turns'])))
    print(("Current experience replay buffer size %s" % (len(agent.experience_replay_pool))))


"""def save_replay_pool_to_csv(experience_replay_pool, csv_path, max_turns=100000):
    # check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # count how many lines already written
    if file_exists:
        with open(csv_path, 'r') as f:
            existing_lines = sum(1 for _ in f)
    else:
        existing_lines = 0
    
    # calculate how many new can be added
    remaining_slots = max_turns - existing_lines
    if remaining_slots <= 0:
        print("CSV already has 100000 turns. No more data will be added.")
        return
    
    # prepare data to write
    rows_to_write = []
    for experience in experience_replay_pool:
        # assuming each experience is (state, action, reward, next_state, done)
        state, action, reward, next_state, done = experience
        rows_to_write.append([state, action, reward, next_state, done])
    
    # only add up to remaining slots
    rows_to_write = rows_to_write[:remaining_slots]
    
    # write or append
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # if new file, write header first
        if not file_exists:
            writer.writerow(['state', 'action', 'reward', 'next_state', 'done'])
        writer.writerows(rows_to_write)
    
    print(f"Saved {len(rows_to_write)} new experiences to {csv_path}. Now total {existing_lines + len(rows_to_write)}.")

"""



def save_replay_pool_to_csv(experience_replay_pool, csv_path, max_turns=100000, batch_size=10000):
    # Count current rows
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            current_rows = sum(1 for _ in f)
    else:
        current_rows = 0  # no file, no rows

    # Check if CSV is already full
    if current_rows >= max_turns + 1:  # +1 for header
        print(f"CSV already has {current_rows} rows. Skipping write.")
        return

    # Check if buffer has enough samples
    if len(experience_replay_pool) < batch_size:
        print(f"Buffer has only {len(experience_replay_pool)} entries. Waiting until it reaches {batch_size}.")
        return

    # Calculate how many rows can still be written
    rows_written = current_rows - 1 if current_rows > 0 else 0
    rows_remaining = max_turns - rows_written

    if rows_remaining <= 0:
        print(f"CSV has reached {max_turns} = {len(experience_replay_pool)} rows.")
        return

    # Determine if header is needed
    write_header = current_rows == 0

    # Limit batch to remaining space


    rows_to_write = min(batch_size, rows_remaining)
    buffer_list = list(experience_replay_pool)
    data_to_write = buffer_list[:rows_to_write]
    #data_to_write = experience_replay_pool[:rows_to_write]

    # Write to CSV
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(['state', 'action', 'reward', 'next_state', 'done'])
        for exp in data_to_write:
            writer.writerow(exp)

    print(f"Wrote {rows_to_write} rows to {csv_path}.")



        
#returns_f = open('returns2.log', 'w+')
def run_episodes(count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    if (agt == 9 or agt == 12 or agt == 13) and params['trained_model_path'] == None and warm_start == 1:
        print ('warm_start starting ...')
        warm_start_simulation()
        print ('warm_start finished, start RL training ...')
    
    for episode in range(count):
        print(("Episode: %s" % (episode)))
        dialog_manager.initialize_episode()
        episode_over = False
        
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            agent.returns[0].append(reward)
            cumulative_reward += reward
                
            if episode_over:
                if reward > 0:
                    print ("Successful Dialog!")
                    successes += 1
                else: print ("Failed Dialog!")
                
                cumulative_turns += dialog_manager.state_tracker.turn_count
        '''
        for i in reversed(range(1, len(agent.returns[0]))):
            agent.returns[0][i - 1] += agent.returns[0][i] * 0.95
            returns_f.writelines('%f %f\n' % (agent.returns[0][i], agent.returns[1][i]))
        returns_f.writelines('%f %f\n' % (agent.returns[0][0], agent.returns[1][0]))
        print(agent.returns)
        '''
        # simulation
        if (agt == 9 or agt == 12 or agt == 13) and params['trained_model_path'] == None:
            agent.predict_mode = True
            simulation_res = simulation_epoch(simulation_epoch_size, train=True)
            performance_records['ave_intrinsic_reward'][episode] = simulation_res['ave_intrinsic_reward']
            agent.predict_mode = False
            
            simulation_res = simulation_epoch(simulation_epoch_size, train=False)
            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']
            
            if simulation_res['success_rate'] >= best_res['success_rate']:
                if simulation_res['success_rate'] >= success_rate_threshold: # threshold = 0.30
                    #Run below for data collecting
                    """if len(agent.experience_replay_pool)>=10000:
                        save_replay_pool_to_csv(agent.experience_replay_pool, "replay_data_resoho.csv")"""
                    agent.reset_replay()
                    #agent.experience_replay_pool = deque(maxlen=params['experience_replay_pool_size']) 
                    agent.predict_mode = True
                    simulation_epoch(simulation_epoch_size, train=False)
                    agent.predict_mode = False
                    
                
            """if best_res['success_rate'] >= 0.5 and len(agent.experience_replay_pool)>=10000 :
                #agent.reset_replay()#agent.experience_replay_pool
                save_replay_pool_to_csv(agent.experience_replay_pool, "replay_data_res.csv")"""



            if simulation_res['success_rate'] > best_res['success_rate']:
                best_model['model'] = copy.deepcopy(agent)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['epoch'] = episode


                
            
            print(("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (performance_records['success_rate'][episode], performance_records['ave_reward'][episode], performance_records['ave_turns'][episode], best_res['success_rate'])))
            print(('intrinsic reward: %s' %(performance_records['ave_intrinsic_reward'][episode])))
            if episode % save_check_point == 0 and params['trained_model_path'] == None: # save the model every 10 episodes
                save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], episode)
                save_performance_records(params['write_model_dir'], agt, performance_records)
        
        print(("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (episode+1, count, successes, episode+1, float(cumulative_reward)/(episode+1), float(cumulative_turns)/(episode+1))))
        print(('Epsilon: %.4f' % agent.epsilon))
        agent.epsilon *= 0.95
    print(("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (successes, count, float(cumulative_reward)/count, float(cumulative_turns)/count)))
    status['successes'] += successes
    status['count'] += count
    
    if (agt == 9 or agt == 12 or agt == 13)  and params['trained_model_path'] == None:
        save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], count)
        save_performance_records(params['write_model_dir'], agt, performance_records)
    
    
run_episodes(num_episodes, status)
