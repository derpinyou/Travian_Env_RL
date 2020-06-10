from make_env import village_info_dict_of_dicts, building_info, TravianEnv, np
import tqdm
rewards_storage = []
A = TravianEnv(village_info_dict_of_dicts, building_info, 10)
A.reset()
for i in tqdm.tqdm(range(10)):

    while A.current_time <= 2592000:
        w = np.random.choice(range(23))
        print(A.step(w))
    rewards_storage.append(A.Total_r)
    A.reset()
print(rewards_storage)
print(sum(rewards_storage)/1000)
print(max(rewards_storage))
print(min(rewards_storage))
