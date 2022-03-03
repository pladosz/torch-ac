from multiprocessing import Process, Pipe
import gym
import numpy as np
import time

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                env.seed(140)
                obs = env.reset()
                env.seed(140)
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            env.seed(140)
            obs = env.reset()
            env.seed(140)
            conn.send(obs)
        elif cmd == "pos":
            pos = env.agent_pos
            rot = env.agent_dir
            pos = np.insert(pos,[0], rot)
            conn.send(pos)
        elif cmd == "close":
            env.close()
            exit()
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            self.processes.append(p)
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results
    
    def get_positions(self):
        for local in self.locals:
            local.send(("pos",None))
        pos = self.envs[0].agent_pos
        rot = self.envs[0].agent_dir
        pos = np.insert(pos,[0], rot)
        positions = [pos] + [local.recv() for local in self.locals]
        return positions
    
    def close(self):
        for local,process in zip(self.locals, self.processes):
            local.send(("close",None))
            local.close()
            process.terminate()
            process.join()

    def render(self):
        raise NotImplementedError