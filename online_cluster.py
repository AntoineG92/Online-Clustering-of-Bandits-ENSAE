import numpy as np
import numpy.random as npr
import numpy.linalg
import pandas as pd
from scipy.spatial.distance import euclidean
import networkx as nx

class OLCB():
    def __init__(self,T,n_user,D,c):
        self.T=T
        self.n_user=n_user
        self.D=D
        self.c=c
        self.V= nx.dense_gnm_random_graph(n_user,3*int(np.log(n_user)*n_user))
        self.m=len( list( nx.connected_component_subgraphs(self.V) ) )  
        self.list_i=[]
        self.list_m=[]
        self.list_C=[]


    #sphere unitaire pour générer les matrices de contexte C à chaque période
    def sphere_unif(self,ndim,npoints):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return (vec)
    #Fonction pouvant servir à contrôler la taille des clusters
    def card_clust(self,z,n,m,j):
        denom=0
        for i in np.arange(1,m+1):
            denom=denom+i**(-z)
        return( n*(j**(-z))/denom )

    def CLUB(self,sigma,alpha,alpha2,z):
        sphere_unif=self.sphere_unif
        card_clust=self.card_clust
        n_user=self.n_user
        D=self.D
        T=self.T
        list_C=self.list_C
        c=self.c
        list_i=self.list_i
        list_m=self.list_m
        V=self.V
        U=sphere_unif(n_user,D)
        regret_cum=np.zeros(T)
        list_payoff=[]
        list_CB=np.zeros(T)
        list_omega=np.zeros(T)
        d_M=dict()
        d_b=dict()
        for i in range(n_user):
            d_M['M%d' % i]=np.identity(D)
            d_b['b%d' % i]=np.zeros(D)
        #matrice U des vrais u_j
        self.U=sphere_unif(n_user,D)
        for cont in range(T):
            list_C.append(sphere_unif(D,c))
        for t in range(T):
            #choisir aléatoirement un user i
            i=int(npr.uniform(0,n_user))
            list_i.append(i)
            #reçoit un vecteur contexte associé au user i
            C = list_C[ int( npr.uniform(0,T) ) ] 
            #On genère omega
            omega=np.zeros([n_user,D])
            omega[i,:]=np.dot(np.linalg.inv(d_M['M'+str(int(i))]),d_b['b'+str(int(i))])
            #on récupère tous les indices qui appartiennent au même cluster que celui de i
            M_index = [ n for n in V if nx.has_path(V,n,i)]
            if M_index==[]:   #si le noeud i est tout seul on ajoute i
                M_index=np.array([i])
            # On somme les matrices M du même cluster et on estime M_bar
            M_sum=sum([d_M['M'+str(int(k))] for k in M_index])
            M_bar=np.identity(D)+M_sum-len(M_index)*np.identity(D)
            b_bar=sum([d_b['b'+str(int(k))] for k in M_index])
            # On calcule omega_bar
            omega_bar=np.dot(np.linalg.inv(M_bar),b_bar)
            #détermine k_t optimal pour cluster j_t(i) --> UCB STRATEGY @cluster level
            vect_k=np.zeros(c)
            for k in range(c):
                CB=alpha*np.sqrt(np.dot(np.dot(C[:,k].T,np.linalg.inv(M_bar)),C[:,k])*np.log(t+1))
                vect_k[k]=CB+np.dot(omega_bar.T,C[:,k])
            k_t=[v for v in range(c) if vect_k[v]==np.max(vect_k)][0]
            #calcule payoffs avec u_j
            a_t=np.dot(U[i,:],C[:,k_t])+npr.uniform(-sigma,sigma,size=1)
            other_payoff= list([ np.dot(U[i,:],C[:,n]) for n in range(c) ])
            if t>0:
                regret_cum[t]=regret_cum[t-1]+a_t-[np.dot(U[i,:],C[:,n]) for n in range(c) if np.dot(U[i,:],C[:,n])==max(other_payoff)][0]
            else:
                regret_cum[t]=a_t-[np.dot(U[i,:],C[:,n]) for n in range(c) if np.dot(U[i,:],C[:,n])==max(other_payoff)][0]
            list_payoff.append(a_t)
            # On update les poids
            d_M['M'+str(int(i))]=d_M['M'+str(int(i))]+np.dot(C[:,k_t],C[:,k_t].T)
            d_b['b'+str(int(i))]=d_b['b'+str(int(i))]+a_t*C[:,k_t]
            # On update les clusters
            T_i=list_i.count(i)-1  #nombre de user i piochés sur les périodes précédentes (exclue période t actuelle)
            #On calcul les bornes de confiance pour les users
            CB_tild=np.zeros(n_user)
            CB_tild[i]=alpha2*np.sqrt((1+np.log(1+T_i))/(1+T_i)) #CB pour i,utile dans la boucle sur les autres users
            # On sélectionne les noeuds du clusters associé à i
            nodes_i=[k for k in V[i]]
            list_CB[t]=CB_tild[i].copy()
            for l in nodes_i: #CB pour les autres users voisins de i
                T_i=list_i.count(l)-1
                if list_i.count(l)==0:
                    T_i=0
                CB_tild[l]=alpha2*np.sqrt((1+np.log(1+T_i))/(1+T_i))
                omega[l,:]=np.dot( np.linalg.inv(d_M['M'+str(l)]) , d_b['b'+str(l)] )
                norm_diff_omega=euclidean(omega[l,:],omega[i,:])
                list_CB[t]=list_CB[t]+CB_tild[l]
                list_omega[t]=list_omega[t]+norm_diff_omega
                #si la distance entre users i et l est grande, on brise le lien
                if (norm_diff_omega > (CB_tild[l] + CB_tild[i]) ):
                    V_test=V.copy()
                    V_test.remove_edge(i,l)
                    if( nx.has_path(V_test,i,l) ):
                        V.remove_edge(i,l)
                    else:
                        # extraie le sous-graphe qui contient i
                        V_copy=[h for h in list(nx.connected_component_subgraphs(V)) if h.has_node(i) ][0]
                        n_user_sub=len(V_copy)
                        V_copy.remove_edge(i,l) # ensuite on casse le lien qui divise ce sous-graphe en 2 clusters
                        m_copy=len( list( nx.connected_component_subgraphs(V_copy) ) )
                        list_card=[ card_clust(z,n_user_sub,m_copy,j) for j in np.arange(1,m_copy+1) ]
                        list_card_V=[ len(c) for c in list(nx.connected_component_subgraphs(V_copy)) ]            
                        diff_card=abs( np.array(sorted(list_card))-np.array(sorted(list_card_V)) )
                        if( np.array_equal(diff_card,np.zeros(m_copy)) | np.array_equal(diff_card,np.ones(m_copy) ) | np.array_equal(diff_card,2*np.ones(m_copy)) | np.array_equal(diff_card,3*np.ones(m_copy))):
                            V.remove_edge(i,l)
            
            m=len( list( nx.connected_component_subgraphs(V) ) )
            list_m.append(m)

        return(list_m,list_CB,list_omega,list_payoff,regret_cum,V)