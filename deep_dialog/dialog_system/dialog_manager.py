"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json, copy, csv
from . import StateTracker
from deep_dialog import dialog_config


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """

    def __init__(self, agent, user, act_set, slot_set, movie_dictionary, success_rate_threshold=0.8):
        self.agent = agent
        self.user = user
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.instrinsic_reward = 0
        self.episode_over = False
        self.success_rate = 0  # Track success rate
        self.success_rate_threshold = success_rate_threshold
        self.experience = []  # To store (s, a, s', r, d) tuples


    def initialize_episode(self):
        """ Refresh state for new dialog """
        
        self.reward = 0
        self.instrinsic_reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action = self.user.initialize_episode()
        self.state_tracker.update(user_action = self.user_action)
        
        if dialog_config.run_mode < 3:
            print ("New episode, user goal:")
            print(json.dumps(self.user.goal, indent=2))
        self.print_function(user_action = self.user_action)
            
        self.agent.initialize_episode()
        
    def update_success_rate(self, dialog_status):
        """ Update success rate based on dialog status """
        if dialog_status == dialog_config.SUCCESS_DIALOG:
            self.success_rate += 1
        self.success_rate /= (self.success_rate + 1)  # Normalize the success rate

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
        
        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################
        self.state = self.state_tracker.get_state_for_agent()
        self.agent_action = self.agent.state_to_action(self.state)
        
        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)
        
        self.agent.add_nl_to_action(self.agent_action) # add NL to Agent Dia_Act
        self.print_function(agent_action = self.agent_action['act_slot_response'])
        
        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
        self.reward = self.reward_function(dialog_status)
        #self.instrinsic_reward = self.agent.get_intrinsic_reward(self.state, self.state_tracker.get_state_for_agent(), self.agent_action)
        
        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if self.episode_over != True:
            self.state_tracker.update(user_action = self.user_action)
            self.print_function(user_action = self.user_action)

        ########################################################################
        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        ########################################################################
        if record_training_data:
            self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over)
        
        return (self.episode_over, self.reward)

    
    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -self.user.max_turn #10
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn #20
        else:
            reward = -1
        return reward
    
    def reward_function_without_penalty(self, dialog_status):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = 0
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn
        else:
            reward = 0
        return reward
    
   

    def save_experience_to_csv(experience, filename="experience.csv"):
        """Save (s, a, s', r, d) tuples to a CSV file."""
        fieldnames = ['s', 'a', 's_prime', 'r', 'd']
        
        # Check if the file already exists, if not, write header
        file_exists = False
        try:
            with open(filename, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header if the file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            for exp in experience:
                writer.writerow({
                    's': exp[0],
                    'a': exp[1],
                    's_prime': exp[2],
                    'r': exp[3],
                    'd': exp[4]
                })

    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """
            
        if agent_action:
            if dialog_config.run_mode == 0:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print(("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl'])))
            elif dialog_config.run_mode == 1:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print(("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots'])))
            elif dialog_config.run_mode == 2: # debug mode
                print(("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots'])))
                print(("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl'])))
            
            if dialog_config.auto_suggest == 1:
                print(('(Suggested Values: %s)' % (self.state_tracker.get_suggest_slots_values(agent_action['request_slots']))))
              
        elif user_action:
            if dialog_config.run_mode == 0:
                print(("Turn %d usr: %s" % (user_action['turn'], user_action['nl'])))
            elif dialog_config.run_mode == 1: 
                print(("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots'])))
            elif dialog_config.run_mode == 2: # debug mode, show both
                print(("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots'])))
                print(("Turn %d usr: %s" % (user_action['turn'], user_action['nl'])))
            
            if self.agent.__class__.__name__ == 'AgentCmd': # command line agent
                user_request_slots = user_action['request_slots']
                if 'ticket'in list(user_request_slots.keys()): del user_request_slots['ticket']
                
                if 'reservation' in list(user_request_slots.keys()): del user_request_slots['reservation']
                if 'taxi' in list(user_request_slots.keys()): del user_request_slots['taxi']
                
                if len(user_request_slots) > 0:
                    possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
                    for slot in list(possible_values.keys()):
                        if len(possible_values[slot]) > 0:
                            print(('(Suggested Values: %s: %s)' % (slot, possible_values[slot])))
                        elif len(possible_values[slot]) == 0:
                            print(('(Suggested Values: there is no available %s)' % (slot)))
                else:
                    pass
                  
                    #kb_results = self.state_tracker.get_current_kb_results()
                    #print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))
          
