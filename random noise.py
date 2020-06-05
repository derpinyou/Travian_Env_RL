from make_env import village_info_dict_of_dicts, building_info, TravianEnv, np



rewards_storage = []
A = TravianEnv(village_info_dict_of_dicts, building_info, 10)
for i in range(1000):
    while A.current_time <= 5184000:
        w = np.random.choice(range(23))
        A.step(w)
        if A.res_growths[0][0] == 1 or A.res_growths[0][0] == 0:
            print(w, A.X, A.res_growths, A.gold)
            break
    rewards_storage.append(A.Total_r)
    A.reset()
print(rewards_storage)
print(sum(rewards_storage)/1000)
