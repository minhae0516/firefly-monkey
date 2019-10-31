import pandas as pd
import torch

def data_reader(filename):


    df0 = pd.read_csv(filename)

    a_traj = []
    obs_traj = []
    x_traj =[]

    for i in range(df0.episode.values[-1]):
        df1 = df0[df0.episode == (i+1)]

        v = torch.FloatTensor(df1.ds_v.values).view(-1)
        w = torch.FloatTensor(df1.ds_w.values).view(-1)

        rel_x = torch.FloatTensor(df1.ds_rel_x.values).view(-1)
        rel_y = torch.FloatTensor(df1.ds_rel_y.values).view(-1)
        rel_ang = torch.FloatTensor(df1.ds_rel_ang.values).view(-1)

        del df1




        a_traj_ep = torch.stack([v, w], dim = 1)
        #obs_traj_ep = torch.stack((torch.split(v,1), torch.split(w,1)), dim = 1)
        obs_traj_ep = torch.stack((v.view(-1,1), w.view(-1,1)), dim=1)
        x_traj_ep = torch.stack([rel_x, rel_y, rel_ang, v, w], dim = 1)

        a_traj_ = torch.split(a_traj_ep, 1)
        #obs_traj_ = torch.split(obs_traj_ep, 1)
        obs_traj_ = obs_traj_ep
        x_traj_ = torch.split(x_traj_ep, 1)

        x_traj.append(x_traj_)
        obs_traj.append(obs_traj_)
        a_traj.append(a_traj_)


    return a_traj, obs_traj, x_traj




