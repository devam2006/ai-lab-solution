import unittest
import numpy as np
import torch
from warehouse import Warehouse, q_update, targets, Network

class Tests(unittest.TestCase):
    def test_single(self):
        env = Warehouse()
        for state, action, expected, reward, term in [((0,0),0,(0,0),-1,False), ((1,1),3,(1,1),-1,False), ((2,1),3,(2,2),-.1,False), ((2,1),4,(2,1),-.1,False), ((2,3),3,(2,4),10,True)]:
            env.reset(state); nxt,r,t,tr,_ = env.step(action)
            self.assertEqual((nxt,r,t,tr), (expected,reward,term,False))
        env.reset((0,0))
        for i in range(40): nxt,r,t,tr,_ = env.step(4)
        self.assertTrue(tr); self.assertFalse(t); self.assertEqual(nxt,(0,0))

    def test_q_update(self):
        for terminated, target, updated in [(False,2.9,.9),(True,5,1.32),(False,2.9,.9)]:
            q = np.zeros((25,5)); q[0,0] = .4; q[1,2] = 2
            self.assertAlmostEqual(q_update(q,(0,0),0,5 if terminated else 1,(0,1),terminated),target)
            self.assertAlmostEqual(q[0,0],updated)

    def test_dqn_targets_and_gradients(self):
        net = Network(25,5); target = Network(25,5)
        values = torch.tensor([[0.,2.,1.,0.,0.]] * 3, requires_grad=True)
        with torch.no_grad(): y = targets(torch.tensor([1.,5.,1.]),torch.tensor([False,True,False]),values)
        self.assertTrue(torch.allclose(y,torch.tensor([2.9,5.,2.9])))
        states = torch.eye(25)[:3]; actions = torch.tensor([1,3,4]); output = net(states)
        pred = output.gather(1,actions[:,None]).squeeze(1)
        self.assertTrue(torch.equal(pred,torch.stack([output[i,actions[i]] for i in range(3)])))
        with torch.no_grad(): detached = targets(torch.ones(3), torch.tensor([False,True,False]), target(states))
        torch.nn.functional.mse_loss(pred,y + detached).backward()
        self.assertIsNone(values.grad); self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_joint(self):
        env = Warehouse(True)
        cases = [(((2,1),(2,3)),3,2,((2,1),(2,3)),-10,True), (((2,1),(2,2)),3,2,((2,1),(2,2)),-10,True), (((2,1),(2,2)),3,3,((2,2),(2,3)),-.1,False), (((2,1),(2,2)),0,4,((1,1),(2,2)),-.1,False)]
        for state,a,b,nxt,reward,term in cases:
            env.reset(state); s,r,t,tr,info=env.step(5*a+b)
            self.assertEqual((s,r,t,tr),(nxt,reward,term,False))
        env.reset(((2,4),(1,1))); s,r,t,tr,info=env.step(3*5+4)
        self.assertEqual(s,((2,4),(1,1))); self.assertEqual(r,-.1)
        s,r,t,tr,info=env.step(3*5+4); self.assertEqual(r,-.1)
        env.reset(((2,3),(2,1))); s,r,t,tr,info=env.step(3*5+2)
        self.assertEqual(r,14.9); self.assertTrue(t); self.assertTrue(info['success'])
        env.reset(((2,1),(2,2))); s,r,t,tr,info=env.step(3*5+4)
        self.assertTrue(info['collision'])

if __name__ == '__main__': unittest.main(verbosity=2)
