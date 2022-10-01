import numpy as np
from scipy.stats import poisson
from scipy.stats import hypergeom
from scipy.stats import beta
from scipy.stats import binom


class SIS(object):
   
    
    def __init__(self,N,trans):   #kappa is the poisson rate of the new tests arrival
                                                        #trans is the state transition matrix 
        self.discount=1
        self.states=np.arange(N+1)
        self.transition=trans
        
    @property
    def num_states(self):
        return len(self.states)

    @property
    def num_actions(self):
        return len(self.actions)
    '''
    def update_belief(self,c_states,c_belief, action,obs):     #update belief by rejective sampling
        n=len(self.states)
        new_states=[]
        while len(new_states)<n:
            idx=np.random.choice(np.linspace(0,len(self.states)-1,len(self.states)),p=c_belief)
            s_state=c_states[int(idx)]
            n_state=self.next_state(s_state,action,obs)
            if self.gen_observation(n_state,action)[0]==obs[0]:
                    new_states.append(n_state)
        #print(new_states)
        bel=np.ones(len(self.states))
        for i in range(len(self.states)):
            #print(self.get_g(new_states[i]))
            bel[i]=self.observation_function(action, new_states[i], obs)
            #print(bel[i])
        new_belief=bel/np.sum(bel)
        #print(new_belief)
        return new_states,new_belief
    '''
       
    def actions(self):
        return np.arange(self.stock+1)
            
    def observations(self):
        return self.actions
       
    def stock_update(self,action,new_test):
        self.stock+=new_tests
        
    def observation_function(self, action, state, obs):
        return hypergeom.pmf(obs,self.num_states-1,state,action)
       

    def transition_function(self,si,sj):
        return self.transition[si][sj]
                    

    def reward_vector(self,b):
        center=np.ones(len(b))/len(b)
        grad=np.ones(len(b))
        dsc=np.sum((b-center)**2)**(1/2)
        if dsc==0:
            v=np.zeros(len(b)+1)
        else:
            for i in range(len(b)):
                grad[i]=round(b[i]/dsc,4)
            intercept=round(dsc-np.dot(b,grad),4)
            v=np.append(grad,intercept)
            
        return v
 

    def cost_function(self, action):
        return 0
           

    def take_action(self,action):
        """
        Accepts an action and changes the underlying environment state
        
        action: action to take
        return: next state, observation and reward
        """
        #print(self.states)
        #print(self.curr_state)
        n_states,n_belief,state, observation, reward,cost= self.simulate_action(self.states,self.belief,self.curr_state,action)
        self.states=n_states
        self.belief=n_belief
        self.stock-=action
        #print(self.curr_state)
        return state, observation, reward,cost

  