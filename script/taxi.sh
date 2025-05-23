for ((i=0;i<5;i=i+1))
do
	save_path="./taxi/dqn/$i"	
	mkdir -p $save_path

	python run.py \
	--agt 13 \
	--usr 3 \
	--max_turn 30 \
	--kb_path ./deep_dialog/data_taxi/taxi.kb.2k.v1.p \
	--goal_file_path ./deep_dialog/data_taxi/user_goals_first.v4.p \
	--slot_set ./deep_dialog/data_taxi/taxi_slots.txt \
	--act_set ./deep_dialog/data_taxi/dia_acts.txt \
	--dict_path ./deep_dialog/data_taxi/slot_dict.v1.p \
	--nlg_model_path ./deep_dialog/models/nlg/taxi/lstm_tanh_[1532457558.95]_95_99_194_0.985.p \
	--nlu_model_path ./deep_dialog/models/nlu/taxi/lstm_[1532583523.63]_88_99_400_0.998.p \
	--diaact_nl_pairs ./deep_dialog/data_taxi/sim_dia_act_nl_pairs.json \
	--dqn_hidden_size 80 \
	--experience_replay_pool_size 10000 \
	--episodes 2000 \
	--simulation_epoch_size 100 \
	--write_model_dir $save_path \
	--run_mode 3 \
	--act_level 0 \
	--slot_err_prob 0.00 \
	--intent_err_prob 0.00 \
	--batch_size 16 \
	--warm_start 1 \
	--warm_start_epochs 120 \
	--epsilon 0.00 \
	--gamma 0.95 \
	--dueling_dqn 1 \
	--double_dqn 0 \
	--icm 0 \
	--per 0 \
	--noisy 1 \
	--distributional 1
done
