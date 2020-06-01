import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

building_names = ['granary', 'storage', 'main', 'farm', 'pit', 'mine', 'lumber']
building_list = []
for name in building_names:
    building_info = []
    text_file = open(name + ".txt", "r")
    lines = text_file.readlines()
    for line in lines:
        line = line[:-1].split('\t')
        line_ok = []
        if line == ['']:
            continue
        else:
            for j in range(len(line)):
                if j != 5:
                    line_ok.append(int(line[j]))
                else:
                    l = line[j].split(':')
                    to_append = 3600*int(l[0]) + 60*int(l[1]) + int(l[2])
                    line_ok.append(to_append)
        building_info.append(line_ok)
    building_list.append(building_info)
dict_to_do = {building_names[j]: building_list[j] for j in range(len(building_names))}
building_info = {building: {j+1: {'costs': dict_to_do.get(building)[j][1:6],
                                  'gains': dict_to_do.get(building)[j][6:8],
                                  'special': dict_to_do.get(building)[j][8]} for j in range(20)} for building
                                   in building_names}

#village_n = int(input())   # сколько деревнь?
#village_info_list = []
#for j in range(village_n):
#    village_info_list.append([int(x) for x in input().split(' ')]) ### ффффффккккллллрррргсазгдж + время + н/к
#keys = ['village' + str(j) for j in range(village_n)]
#village_info_dict_of_dicts = {}
#for j in range(len(keys)):
#    to_add = {'farm': village_info_list[j][:6],
#              'mine': village_info_list[j][6:10],
#              'lumber': village_info_list[j][10:14],
#              'pit': village_info_list[j][14:18],
#              'inside': village_info_list[j][18:21],
#              'resources': village_info_list[j][21:25],
#              'time_remaining': village_info_list[j][25],
#              'gains': village_info_list[j][26:],
#              'waiting_for': 'nothing'
#    }
#    village_info_dict_of_dicts[keys[j]] = to_add

### пример waiting_for - ('farm', 1, 6)
### inside[0] амбар inside[1] склад inside[2] гз

village_info_dict_of_dicts = {'village0': {'farm': [5, 5, 5, 5, 5, 5], 'mine': [5, 5, 5, 5],
                              'lumber': [5, 5, 5, 5], 'pit': [5, 5, 5, 5], 'inside': [5, 5, 5],
                              'resources': [800, 800, 800, 800], 'time_remaining': 0,
                              'gains': [100, 200], 'waiting_for' : 'nothing'}}


class TravianEnv(gym.Env):

    def __init__(self, X, buildings_info, gold):
        self.X = X
        self.village_n = len(X)
        self.buildings_info = buildings_info
        self.gold = gold
        self.action_space = spaces.Discrete(22 * self.village_n + 1)
        self.current_time = 0
        self.granary_capacities = [self.current_capacity_and_boost(i)[0] for i in range(self.village_n)]
        self.storage_capacities = [self.current_capacity_and_boost(i)[1] for i in range(self.village_n)]
        self.boost = [self.current_capacity_and_boost(i)[2] for i in range(self.village_n)]
        self.res_growths = [self.res_growth(i) for i in range(self.village_n)]

    def current_capacity_and_boost(self, vil_n):
        """
        узнать вместимость амбара, склада по номеру деревни и буст от главного здания
        :param self:
        :param vil_n: количество деревнь
        :return: список [вместимость амбара, вместимость склада, буст%]
        """
        granary_lvl = self.X['village' + str(vil_n)].get('inside')[0]
        storage_lvl = self.X['village' + str(vil_n)].get('inside')[1]
        boost_lvl = self.X['village' + str(vil_n)].get('inside')[2]
        granary_capacity = self.buildings_info['granary'].get(granary_lvl).get('special')
        storage_capacity = self.buildings_info['storage'].get(storage_lvl).get('special')
        boost = self.buildings_info['main'].get(boost_lvl).get('special')
        return [granary_capacity, storage_capacity, boost]

    def res_growth(self, vil_n):
        farm_growth = 0
        for farm in self.X['village' + str(vil_n)].get('farm'):
            f = self.buildings_info['farm'].get(farm).get('special')
            farm_growth += f
        mine_growth = 0
        for mine in self.X['village' + str(vil_n)].get('mine'):
            m = self.buildings_info['mine'].get(mine).get('special')
            mine_growth += m
        lumber_growth = 0
        for lumber in self.X['village' + str(vil_n)].get('lumber'):
            lu = self.buildings_info['lumber'].get(lumber).get('special')
            lumber_growth += lu
        pit_growth = 0
        for pit in self.X['village' + str(vil_n)].get('pit'):
            p = self.buildings_info['pit'].get(pit).get('special')
            pit_growth += p
        return [farm_growth, mine_growth, lumber_growth, pit_growth]

    def count_time_pace(self):
        """
        будем настраивать, сколько времени "ждёт" агент, если выбирает действие ждать
        это будет минимум(завершение ближайшей постройки; ещё одна постройка станет доступной к строительству),
        если везде постройки не завершены, и это будет минимум до ближайшей постройки, которая станет доступной,
        если хотя бы в одной деревне time_remaining == 0.
        """
        time_remaining_list = [self.X['village' + str(k)].get('time_remaining') for k in range(self.village_n)]
        min_building_time = min(time_remaining_list)
        time = []
        for j in range(self.village_n):
            for a in range(21):
                if a <= 5:
                    farm_lvl = self.X['village' + str(j)].get('farm')[a] + 1
                    resources_in = self.X['village' + str(j)].get('resources')
                    resources_required = self.buildings_info['farm'].get(farm_lvl).get('costs')[:4]
                    if not all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                        wow = [0 if resources_in[m] >= resources_required[m] else resources_required[m] - resources_in[m] for
                             m in range(len(resources_in))]
                        time.append(max([wow[0] / self.res_growth(j)[0], wow[1] / self.res_growth(j)[1],
                                         wow[2] / self.res_growth(j)[2],
                                         wow[3] / self.res_growth(j)[3]]))
                if 5 < a <= 9:
                    mine_lvl = self.X['village' + str(j)].get('mine')[a-6] + 1
                    resources_in = self.X['village' + str(j)].get('resources')
                    resources_required = self.buildings_info['mine'].get(mine_lvl).get('costs')[:4]
                    if not all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                        wow = [0 if resources_in[w] >= resources_required[w] else resources_required[w] - resources_in[w]
                               for
                               w in range(len(resources_in))]
                        time.append(max([wow[0] / self.res_growth(j)[0], wow[1] / self.res_growth(j)[1],
                                         wow[2] / self.res_growth(j)[2],
                                         wow[3] / self.res_growth(j)[3]]))
                if 9 < a <= 13:
                    lumber_lvl = self.X['village' + str(j)].get('lumber')[a-10] + 1
                    resources_in = self.X['village' + str(j)].get('resources')
                    resources_required = self.buildings_info['lumber'].get(lumber_lvl).get('costs')[:4]
                    if not all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                        wow = [0 if resources_in[b] >= resources_required[b] else resources_required[b] - resources_in[b]
                               for
                               b in range(len(resources_in))]
                        time.append(max([wow[0] / self.res_growth(j)[0], wow[1] / self.res_growth(j)[1],
                                         wow[2] / self.res_growth(j)[2],
                                         wow[3] / self.res_growth(j)[3]]))
                if 13 < a <= 17:
                    pit_lvl = self.X['village' + str(j)].get('pit')[a-14] + 1
                    resources_in = self.X['village' + str(j)].get('resources')
                    resources_required = self.buildings_info['pit'].get(pit_lvl).get('costs')[:4]
                    if not all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                        wow = [0 if resources_in[q] >= resources_required[q] else resources_required[q] - resources_in[q]
                               for
                               q in range(len(resources_in))]
                        time.append(max([wow[0] / self.res_growth(j)[0], wow[1] / self.res_growth(j)[1],
                                         wow[2] / self.res_growth(j)[2],
                                         wow[3] / self.res_growth(j)[3]]))
                if 17 < a <= 20:
                    inside_building = ['granary', 'storage', 'main'][a - 18]
                    inside_lvl = self.X['village' + str(j)].get('inside')[a - 18] + 1
                    resources_in = self.X['village' + str(j)].get('resources')
                    resources_required = self.buildings_info[inside_building].get(inside_lvl).get('costs')[:4]
                    if not all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                        wow = [0 if resources_in[e] >= resources_required[e] else resources_required[e] - resources_in[e]
                               for
                               e in range(len(resources_in))]
                        time.append(max([wow[0] / self.res_growth(j)[0], wow[1] / self.res_growth(j)[1],
                                         wow[2] / self.res_growth(j)[2],
                                         wow[3] / self.res_growth(j)[3]]))
        if min_building_time != 0:
            return min_building_time
        else:
            if len(time) == 0:
                return 0
            else:
                return 3600*min(time)

    def is_available_and_rr(self, action_):
        """
        проверяем, не строится ли что-то ещё и есть ли ресурсы
        :param action_:
        :return: доступно ли, сколько ресурсов надо, сколько времени занимает
        """
        excess_ = action_ % 22
        integer_ = action_ // 22
        what_to_change = 0

        if action_ == 22 * self.village_n:
            available = 1
            resources_required = 0
            time_required = 0
        elif excess_ == 21 and self.X['village' + str(integer_)].get('time_remaining') != 0:
            time_required = 0
            available = 1
            if self.X['village' + str(integer_)].get('time_remaining') <= 7200:
                resources_required = 1
                if self.gold == 0:
                    available = 0
            else:
                resources_required = 2
                if self.gold < 2:
                    available = 0
        elif self.X['village' + str(integer_)].get('time_remaining') > 0:
            available = 0
            resources_required = 0
            time_required = 0
        else:
            available = 0
            resources_required = 0
            time_required = 0
            if excess_ <= 5:
                farm_lvl = self.X['village' + str(integer_)].get('farm')[excess_] + 1
                resources_in = self.X['village' + str(integer_)].get('resources')
                time_required = self.buildings_info['farm'].get(farm_lvl).get('costs')[4]
                resources_required = self.buildings_info['farm'].get(farm_lvl).get('costs')[:4]
                if all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                    tc = self.buildings_info['farm'].get(farm_lvl).get('gains')[0]
                    if self.X['village' + str(integer_)].get('gains')[0] + 2 + tc <= self.res_growth(integer_)[0]:
                        available = 1
                what_to_change = ['farm', excess_]
            elif 5 < excess_ <= 9:
                mine_lvl = self.X['village' + str(integer_)].get('mine')[excess_ - 6] + 1
                resources_in = self.X['village' + str(integer_)].get('resources')
                resources_required = self.buildings_info['mine'].get(mine_lvl).get('costs')[:4]
                time_required = self.buildings_info['mine'].get(mine_lvl).get('costs')[4]
                if all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                    tc = self.buildings_info['mine'].get(mine_lvl).get('gains')[0]
                    if self.X['village' + str(integer_)].get('gains')[0] + 2 + tc <= self.res_growth(integer_)[0]:
                        available = 1
                what_to_change = ['mine', excess_-6]
            elif 9 < excess_ <= 13:
                lumber_lvl = self.X['village' + str(integer_)].get('lumber')[excess_ - 10] + 1
                resources_in = self.X['village' + str(integer_)].get('resources')
                resources_required = self.buildings_info['lumber'].get(lumber_lvl).get('costs')[:4]
                time_required = self.buildings_info['lumber'].get(lumber_lvl).get('costs')[4]
                if all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                    tc = self.buildings_info['lumber'].get(lumber_lvl).get('gains')[0]
                    if self.X['village' + str(integer_)].get('gains')[0] + 2 + tc <= self.res_growth(integer_)[0]:
                        available = 1
                what_to_change = ['lumber', excess_ - 10]
            elif 13 < excess_ <= 17:
                pit_lvl = self.X['village' + str(integer_)].get('pit')[excess_ - 14] + 1
                resources_in = self.X['village' + str(integer_)].get('resources')
                resources_required = self.buildings_info['pit'].get(pit_lvl).get('costs')[:4]
                time_required = self.buildings_info['pit'].get(pit_lvl).get('costs')[4]
                if all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                    tc = self.buildings_info['pit'].get(pit_lvl).get('gains')[0]
                    if self.X['village' + str(integer_)].get('gains')[0] + 2 + tc <= self.res_growth(integer_)[0]:
                        available = 1
                what_to_change = ['pit', excess_ - 14]
            elif 17 < excess_ <= 20:
                inside_building = ['granary', 'storage', 'main'][excess_ - 18]
                inside_lvl = self.X['village' + str(integer_)].get('inside')[excess_ - 18] + 1
                resources_in = self.X['village' + str(integer_)].get('resources')
                resources_required = self.buildings_info[inside_building].get(inside_lvl).get('costs')[:4]
                time_required = self.buildings_info[inside_building].get(inside_lvl).get('costs')[4]
                if all(resources_in[i] >= resources_required[i] for i in range(len(resources_in))):
                    tc = self.buildings_info[inside_building].get(inside_lvl).get('gains')[0]
                    if self.X['village' + str(integer_)].get('gains')[0] + 2 + tc <= self.res_growth(integer_)[0]:
                        available = 1
                what_to_change = ['inside', excess_ - 18]

        return available == 1, resources_required, time_required, what_to_change

    def step(self, action):

        reward = 0


        excess_ = action % 22
        integer_ = action // 22

        if self.is_available_and_rr(action)[0]:
            if action == 22 * self.village_n:
                step = self.count_time_pace()
                for vil in range(self.village_n):
                    time_ = self.X['village' + str(vil)]['time_remaining']
                    resources_change = []
                    res_lists = (self.granary_capacities[vil], self.storage_capacities[vil],
                                 self.storage_capacities[vil], self.storage_capacities[vil])
                    for p in range(4):
                        resources_change.append(min(self.X['village' + str(vil)]['resources'][p]
                                                    + self.res_growths[vil][p] * step / 3600,
                                                    res_lists[p]))
                    self.X['village' + str(vil)]['resources'] = resources_change

                    time = max(0, self.X['village' + str(vil)]['time_remaining'] - self.count_time_pace())
                    self.X['village' + str(vil)]['time_remaining'] = time

                    if self.X['village' + str(vil)]['time_remaining'] == 0 and self.X['village' + str(vil)][
                        'waiting_for'] != 'nothing':
                        wait_for = self.X['village' + str(vil)]['waiting_for']
                        print(wait_for)
                        self.X['village' + str(vil)][wait_for[0]][wait_for[1]] = wait_for[2]
                        self.X['village' + str(vil)]['waiting_for'] = 'nothing'
                        b_lvl = self.X['village' + str(vil)][wait_for[0]][wait_for[1]]
                        if wait_for[0] == 'inside':
                            name_for_info = ['granary', 'storage', 'main'][wait_for[1]]
                        else:
                            name_for_info = wait_for[0]
                        self.X['village' + str(vil)]['gains'][0] += self.buildings_info[name_for_info][b_lvl]['gains'][
                            0]
                        self.X['village' + str(vil)]['gains'][1] += self.buildings_info[name_for_info][b_lvl]['gains'][
                            1]

                self.current_time += step
                reward = - step * 10 ** (-5)
            else:
                if excess_ == 21:
                    print(self.X['village' + str(integer_)]['waiting_for'])
                    self.gold -= self.is_available_and_rr(action)[1]
                    wait_for = self.X['village' + str(integer_)]['waiting_for']
                    b_lvl = self.X['village' + str(integer_)][wait_for[0]][wait_for[1]]
                    if wait_for[0] == 'inside':
                        name_for_info = ['granary', 'storage', 'main'][wait_for[1]]
                        self.X['village' + str(integer_)]['gains'][0] += \
                        self.buildings_info[name_for_info][b_lvl]['gains'][
                            0]
                        self.X['village' + str(integer_)]['gains'][1] += \
                        self.buildings_info[name_for_info][b_lvl]['gains'][
                            1]
                    else:
                        name_for_info = wait_for[0]
                        self.X['village' + str(integer_)]['gains'][0] += self.buildings_info[name_for_info][b_lvl]['gains'][
                            0]
                        self.X['village' + str(integer_)]['gains'][1] += self.buildings_info[name_for_info][b_lvl]['gains'][
                            1]
                    self.X['village' + str(integer_)][wait_for[0]][wait_for[1]] = wait_for[2]
                    self.X['village'+str(integer_)]['waiting_for'] = 'nothing'
                    self.X['village' + str(integer_)]['time_remaining'] = 0
                else:
                    changes = self.is_available_and_rr(action)[3]
                    res_ = self.X['village' + str(integer_)]['resources']
                    j_costs = self.is_available_and_rr(action)[1]
                    self.X['village' + str(integer_)]['time_remaining'] = self.is_available_and_rr(action)[2]
                    changes.append(self.X['village' + str(integer_)][changes[0]][changes[1]] + 1)
                    self.X['village' + str(integer_)]['waiting_for'] = changes
                    print(self.X['village' + str(integer_)]['waiting_for'])
                    self.X['village' + str(integer_)]['resources'] = [res_[q] - j_costs[q] for q in range(4)]
                    if changes[0] == 'inside':
                        which_one = ['granary', 'storage', 'main'][changes[1]]
                        reward = (self.buildings_info[which_one][changes[2]]['gains'][1] +
                                  self.buildings_info[which_one][changes[2]]['gains'][0]) / 10
                    else:
                        reward = (self.buildings_info[changes[0]][changes[2]]['gains'][1] +
                                  self.buildings_info[changes[0]][changes[2]]['gains'][0])/10
                    print(self.X['village' + str(integer_)]['waiting_for'])
        else:
            pass

        self.granary_capacities = [self.current_capacity_and_boost(i)[0] for i in range(self.village_n)]
        self.storage_capacities = [self.current_capacity_and_boost(i)[1] for i in range(self.village_n)]
        self.boost = [self.current_capacity_and_boost(i)[2] for i in range(self.village_n)]
        self.res_growths = [self.res_growth(i) for i in range(self.village_n)]

        obs = self.X, self.gold, self.storage_capacities, self.granary_capacities,\
              self.boost, self.res_growths, self.current_time

        if self.current_time < 1000000000:
            done = False
        else:
            done = True

        return obs, reward, done, {}

    def reset(self):
        return village_info_dict_of_dicts, 10, [4000], [4000], [86], [198, 132, 132, 132], 0


A = TravianEnv(village_info_dict_of_dicts, building_info, 10)
