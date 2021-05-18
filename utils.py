import numpy as np
import random
from rdkit import Chem
import tensorflow as tf
from sklearn.model_selection import KFold

def shuffle_two_list(list1, list2):
    list_total = list(zip(list1,list2))
    random.shuffle(list_total)
    list1, list2 = zip(*list_total)
    return list1, list2



def load_input_HIV():
    smi_list = []
    prop_list = []
    # Actives
    f = open('./data/TOX/NIHSUH.txt', 'r')
    lines = f.readlines()
    num=0
    for l in lines:
        num += 1
        try:
            smi = l.split(',')[0]
            prop = float(l.split(',')[1])
            smi_list.append(smi)
            prop_list.append(prop)
        except:
            print("Error at line number:",num)
    return smi_list, prop_list




def load_input_tox21(tox_name, max_atoms):
    f = open('./data/tox21/'+tox_name+'_all.txt', 'r')
    lines = f.readlines()
    smi_list = []
    prop_list = []
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        else:
            if( m.GetNumAtoms() < max_atoms+1 ):
                smi_list.append(smi)
                prop_list.append(prop)
    return smi_list, prop_list

def split_train_eval_test(input_list, train_ratio, test_ratio, eval_ratio):
    #no randomization
    num_total = len(input_list)
    num_test = int(num_total*test_ratio)
    num_train = num_total-num_test
    num_eval = int(num_train*eval_ratio)
    num_train -= num_eval
    #here specifies the order of the data
    train_list = input_list[:num_train]
    eval_list = input_list[num_train:num_train+num_eval]
    test_list = input_list[num_train+num_eval:]
    return train_list, eval_list, test_list


def convert_to_graph(smiles_list, max_atoms):
    adj = []
    features = []
    atomlist=[]
    molList=[]
    count = 0
    for i in smiles_list:
        # Mol

        iMol = Chem.MolFromSmiles(i.strip())
        #print("this is i: ",i)
        # print(iMol)

        # if(iMol==None):
        #     count=count+1
        #     print(count)
        #     print(i)
        #     continue

        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        #print("this is iAdjTmp",iAdjTmp)
        if( iAdjTmp.shape[0] <= max_atoms):
            # Feature-preprocessing
            iFeature = np.zeros((max_atoms, 61))
            iFeatureTmp = []

            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only


            iFeature[0:len(iFeatureTmp), 0:61] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            molList.append(iMol)
            # Adj-preprocessing
            iAdj = np.zeros((max_atoms, max_atoms))
            #print("this is iFeatureTmp",(len(iFeatureTmp)))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), 1))
            #print("length of adj_k", len((np.asarray(iAdj))))
    features = np.asarray(features)
    adj = np.asarray(adj)
    #print("length of adj",len(adj))
    return adj, features

def adj_k(adj, k):
    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)
    #print("ret shape",len(ret))
    return convert_adj(ret)

def convert_adj(adj):

    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))

    return d

def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',

                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',

                                       'V', 'Sb', 'Sn', 'Ag', 'Co', 'Se', 'Ti', 'Zn',

                                       'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Hg', 'Pb','Cl']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [0, 1, 2, 3, 4, 5])+

                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)


def one_of_k_encoding(x, allowable_set):

    if x not in allowable_set:

        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    #print list((map(lambda s: x == s, allowable_set)))

    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
