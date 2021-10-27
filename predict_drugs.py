import pandas as pd
import numpy as np

import torch as th
import torch.nn.functional as fn

from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt

from score_functions import *


gamma = 12.0
folder = '/Users/yuhou/Documents/Knowledge_Graph/CBKH/Case_Study/DDi_DG_DiG_GG/'
folder = 'EmbeddingResult1/'
# folder = '/Users/yuhou/Documents/Knowledge_Graph/CBKH/Case_Study/TransE_l2_CBKH_4_whole/'
result_folder = '/Users/yuhou/Documents/Knowledge_Graph/CBKH/Case_Study/result/'
kg_folder = '/Users/yuhou/Documents/Knowledge_Graph/CBKH/Result/'


def transE_l2(head, rel, tail):
    score = head + rel - tail
    return gamma - th.norm(score, p=2, dim=-1)


def predict_candidate_drugs(trail_status):
    entity_df = pd.read_table(folder + 'entities.tsv', header=None)
    entity_df = entity_df.dropna().reset_index(drop=True)
    # approved_drug_df = pd.read_csv(result_folder + 'candidate_drugs_' + trail_status + '.csv')
    approved_drug_df = pd.read_csv('candidate_drugs_approve.csv')
    approved_drug_list = list(approved_drug_df['Drug'])
    print(len(approved_drug_list))

    entity_map = {}
    entity_id_map = {}
    relation_map = {}
    drug_ids = []
    drug_names = []
    disease_ids = []

    for i in range(len(entity_df)):
        entity_id = entity_df.loc[i, 0]
        entity_name = entity_df.loc[i, 1]
        entity_map[entity_name] = int(entity_id)
        entity_id_map[int(entity_id)] = entity_name
        if entity_name.replace('DrugBank:', '') in approved_drug_list:
            drug_ids.append(entity_id)
            drug_names.append(entity_name.replace('DrugBank:', ''))

    # candidate_drugs = approved_drug_df[approved_drug_df['Drug'].isin(drug_names)]
    # candidate_drugs_list = list(candidate_drugs['Drug'])
    # print(len(candidate_drugs_list))

    # candidate_drugs.to_csv(result_folder + 'candidate_drugs_final_DD.csv', index=False)

    disease_vocab = pd.read_csv('disease_vocab.csv')
    AD_related_list = []
    for i in range(len(disease_vocab)):
        primary_id = disease_vocab.loc[i, 'primary']
        disease_name = disease_vocab.loc[i, 'name']
        disease_name = disease_name if not pd.isnull(disease_name) else ''
        if 'alzheimer' in disease_name:
            if primary_id not in AD_related_list:
                AD_related_list.append(primary_id)

    relation_df = pd.read_table(folder + 'relations.tsv', header=None)
    for i in range(len(relation_df)):
        relation_id = relation_df.loc[i, 0]
        relation_name = relation_df.loc[i, 1]
        relation_map[relation_name] = int(relation_id)

    for disease in AD_related_list:
        if disease in entity_map:
            disease_ids.append(entity_map[disease])

    entity_emb = np.load(folder + 'TransR_TONY_3/TONY_TransR_entity.npy')
    rel_emb = np.load(folder + 'TransR_TONY_3/TONY_TransR_relation.npy')
    proj_np = np.load(folder + 'TransR_TONY_3/TONY_TransRprojection.npy')
    proj_emb = th.tensor(proj_np)

    treatment = ['Treats_DDi', 'Palliates_DDi', 'Effect_DDi', 'Associate_DDi', 'Inferred_Relation_DDi',
                 'Semantic_Relation_DDi']
    treatment_rid = [relation_map[treat] for treat in treatment]

    drug_ids = th.tensor(drug_ids).long()
    disease_ids = th.tensor(disease_ids).long()
    treatment_rid = th.tensor(treatment_rid)

    drug_emb = th.tensor(entity_emb[drug_ids])
    treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]

    scores_per_disease = []
    dids = []
    for rid in range(len(treatment_embs)):
        treatment_emb = treatment_embs[rid]
        for disease_id in disease_ids:
            disease_emb = th.tensor(entity_emb[disease_id])
            #score = fn.logsigmoid(transE_l2(drug_emb, treatment_emb, disease_emb))
            score = fn.logsigmoid(transR(drug_emb, treatment_emb, disease_emb, proj_emb, treatment_rid[rid]))
            scores_per_disease.append(score)
            dids.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids)

    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].numpy()
    dids = dids[idx].numpy()

    _, unique_indices = np.unique(dids, return_index=True)
    topk_indices = np.sort(unique_indices)
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]

    candidate_drug_rank = []
    candidate_drug_score = {}
    for i, idx in enumerate(proposed_dids):
        candidate_drug_rank.append(entity_id_map[int(idx)].replace('DrugBank:', ''))
        candidate_drug_score[entity_id_map[int(idx)].replace('DrugBank:', '')] = proposed_scores[i]
    print(len(candidate_drug_rank))
    print(candidate_drug_rank)

    df = pd.DataFrame(columns=['Drug', 'Score'])
    idx = 0
    for drug in candidate_drug_score:
        df.loc[idx] = [drug, candidate_drug_score[drug]]
        idx += 1
    print(df)
    x = np.asarray(df['Score']).reshape(-1, 1)  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df['Score_scaled'] = pd.DataFrame(x_scaled)
    print(df)
    df.to_csv("predict_result_scaled_" + trail_status + ".csv", index=False)
    
if __name__ == '__main__':
    predict_candidate_drugs("0")