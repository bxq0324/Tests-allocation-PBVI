import numpy as np

from solvers import Solver
from util.alpha_vector import AlphaVector
from array import array
from scipy.stats import poisson 
MIN = -np.inf


class PBVI(Solver):
    def __init__(self, model,test_arrival_rate):
        Solver.__init__(self, model)
        self.belief_points = None
        self.alpha_vecs = None
        self.solved = False
        self.kappa=test_arrival_rate
        

    def add_configs(self, belief_points):
        Solver.add_configs(self) 
        self.alpha_vecs=[AlphaVector(a=-1, v=np.zeros(self.model.num_states),z=s) for s in np.arange(220)]
        self.belief_points = belief_points
        self.compute_gamma_reward()

    def compute_gamma_reward(self):
        """
        :return: Action_a => Reward(s,a) matrix
        """
        m=self.model 
        
        rho = np.zeros((len(self.belief_points),m.num_states+1))
        for i,b in enumerate(self.belief_points):
            rho[i]=m.reward_vector(b)
            
        return rho
      

    def compute_gamma_action_obs_eta(self, a, o,stock,eta):
        """
        Computes a set of vectors, one for each previous alpha
        vector that represents the update to that alpha vector
        given an action and observation

        :param a: action index
        :param o: observation index
        """
        m = self.model
        k = self.kappa
        gamma_action_obs_eta = []
        
        for alpha in self.alpha_vecs:
            v = np.zeros(m.num_states)
            if alpha.z==stock-a+eta:
                  
                for i, si in enumerate(m.states):
                    for j, sj in enumerate(m.states):
                        coeff=m.transition_function(si, sj) * m.observation_function(a, sj, o) *poisson.pmf(eta,k) 
                        trunc=round(coeff*alpha.v[j],8)
                        if trunc>0:
                            v[i] += trunc
                    #v[i] *= m.discount
                 
                gamma_action_obs_eta.append(v)
        #print(o,gamma_action_obs_eta)
        return gamma_action_obs_eta

    def solve(self, T):
        if self.solved:
            return

        m = self.model
        '''
        stock=np.zeros(len(self.belief_points))     #an array keeps track of the stock size for each belief points
        for i in range(len(self.belief_points)):
            stock[i]=ini_stock                 
        '''    
        eta_range=15
        for step in range(T):
            print('step:',step)
            # First compute a set of updated vectors for every action/observation pair
            # Action(a) => Observation(o) => UpdateOfAlphaVector (a, o)
            gamma_intermediate={}
            
            for z in range((T-step)*eta_range):
                choices=min(m.num_states,z+1)
                gamma_intermediate[z]={
                     eta:{
                      a:{
                      o: self.compute_gamma_action_obs_eta(a, o,z,eta)
                       for o in np.arange(a+1)
                         } for a in np.arange(choices)
                          } for eta in np.arange(eta_range+1)}
           
            # Now compute the cross sum
            gamma_action_belief = {}
            for z in range((T-step)*eta_range):
                choices=min(m.num_states,z+1)
                gamma_action_belief[z]={}
                rho=self.compute_gamma_reward()
                
                for bidx, b in enumerate(self.belief_points):
                    gamma_action_belief[z][bidx]={}
                    b_=np.append(b,1)
                    max_rho_idx=np.argmax(np.dot(rho,b_))
                   
                    #if z==m.num_states-1:
                        #print(z,bidx,max_rho_idx)
                    for a in np.arange(choices):
                        gamma_action_belief[z][bidx][a] = rho[max_rho_idx].copy()
                        
                        for eta in np.arange(eta_range+1):
                        
                            for o in np.arange(a+1):
                               
                            # only consider the best point
                                #if step==1:
                                    #print('z:',z,'bidx:',bidx,'eta:',eta,'a:',a,'o:',o,np.dot(gamma_intermediate[z][eta][a][o], b))
                                #print(gamma_intermediate[z][eta][a][o])
                                #print('z:',z,'bidx:',bidx,'eta:',eta,'a:',a,'o:',o,np.dot(gamma_intermediate[z][eta][a][o], b))    
                                best_alpha_idx = np.argmax(np.dot(gamma_intermediate[z][eta][a][o], b))
                                best_alpha_=np.append(gamma_intermediate[z][eta][a][o][best_alpha_idx],0)
                                gamma_action_belief[z][bidx][a] += best_alpha_
                        ba=gamma_action_belief[z][bidx][a].copy()       
                        #print(z,bidx,a,ba)
            # Finally compute the new(best) alpha vector set
                   
            self.alpha_vecs, max_val = [], MIN
                                           
            for s in range((T-step)*eta_range):
                                                         
                for bidx, b in enumerate(self.belief_points):
                    best_av, best_aa = None, None
                    b_=np.append(b,1)
                    
                    for a in np.arange(min(m.num_states,s+1)):
                        val = np.dot(gamma_action_belief[s][bidx][a], b_)
                        if best_av is None or val > max_val:
                            max_val = val
                            best_av = gamma_action_belief[s][bidx][a].copy()
                            best_aa = a
                    '''
                    print(np.dot(gamma_action_belief[s][bidx], b_))
                    best_aa=np.argmax(np.dot(gamma_action_belief[s][bidx], b_))
                    best_av==gamma_action_belief[s][bidx][best_aa].copy()
                    '''
                    self.alpha_vecs.append(AlphaVector(a=best_aa, v=best_av,z=s))      
                    
            #if step>=1:                    
            for a in self.alpha_vecs:
                print('best a:',a.action,a.v,'stock:',a.z)
        self.solved = True
                                               

    def get_action(self, belief,stock):
        b_=np.append(belief,1)
        max_v = -np.inf
        best = None
        for av in self.alpha_vecs:
            #print(av.action,av.v)
            if av.z==stock:
                
                v = np.dot(av.v, b_)
                if v > max_v:
                    max_v = v
                    best = av

        return best.action
    
    
    def update_belief(self, belief, action, obs):
        m = self.model
        
        if (a==0)and(o==0):   #no test administered
            b_new=np.matmul(belief,m.trans)
        
        else:
            bel=np.matmul(belief,trans)
            for j,sj in enumerate(m.states):
                p_obs= m.observation_function(action, sj, obs)
                bel[j]*=p_obs
        
        b_new=bel/np.sum(bel)
    
        return b_new
