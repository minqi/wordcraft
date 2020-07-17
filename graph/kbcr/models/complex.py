# -*- coding: utf-8 -*-

import logging

import torch
from torch import nn, Tensor

from .base import BaseLatentFeatureModel

from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ComplEx(BaseLatentFeatureModel):
    def __init__(
        self, num_entities, num_predicates, embedding_size, 
        init_size=1e-3, feature_type='integer', feature_size=300, index2features=None) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_predicates = num_predicates
        self.embedding_size = embedding_size
        self.rank = embedding_size*2

        self.feature_type = feature_type
        self.feature_size = feature_size

        entity_input_size = num_entities if feature_type == 'integer' else feature_size
        sparse = feature_type == 'integer'

        if feature_type == 'integer':
            self.entity_embeddings = nn.Embedding(entity_input_size, self.rank, sparse=True)
            # self.entity_embeddings = nn.Sequential(nn.Embedding(entity_input_size, self.rank, sparse=True), nn.Tanh())
        else:
            self.entity_embeddings = nn.Sequential(
                nn.Linear(entity_input_size, self.rank, bias=True), 
                nn.Tanh())
        self.predicate_embeddings = nn.Embedding(num_predicates, self.rank, sparse=True)
        
        if feature_type == 'integer':
            self.entity_embeddings.weight.data *= init_size
        self.predicate_embeddings.weight.data *= init_size

        if self.feature_type == 'glove':
            assert index2features is not None, 'index2features required for GloVe inputs.'
            self.index2features = index2features

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        arg1 = self.entity_embeddings(arg1)
        arg2 = self.entity_embeddings(arg2)
        rel = self.predicate_embeddings(rel)

        # [B, E]
        rel_real, rel_img = rel[:, :self.embedding_size], rel[:, self.embedding_size:]
        arg1_real, arg1_img = arg1[:, :self.embedding_size], arg1[:, self.embedding_size:]
        arg2_real, arg2_img = arg2[:, :self.embedding_size], arg2[:, self.embedding_size:]

        # [B] Tensor
        score1 = torch.sum(rel_real * arg1_real * arg2_real, 1)
        score2 = torch.sum(rel_real * arg1_img * arg2_img, 1)
        score3 = torch.sum(rel_img * arg1_real * arg2_img, 1)
        score4 = torch.sum(rel_img * arg1_img * arg2_real, 1)

        res = score1 + score2 + score3 - score4

        # [B] Tensor
        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        arg1 = self.entity_embeddings(arg1)
        arg2 = self.entity_embeddings(arg2)
        rel = self.predicate_embeddings(rel)

        if self.feature_type == 'glove':
            emb = self.entity_embeddings(self.index2features)
        else:
            emb = self.entity_embeddings.weight

        rel_real, rel_img = rel[:, :self.embedding_size], rel[:, self.embedding_size:]
        emb_real, emb_img = emb[:, :self.embedding_size], emb[:, self.embedding_size:]

        # [B] Tensor
        score_sp = score_po = None

        if arg1 is not None:
            arg1_real, arg1_img = arg1[:, :self.embedding_size], arg1[:, self.embedding_size:]

            score1_sp = (rel_real * arg1_real) @ emb_real.t()
            score2_sp = (rel_real * arg1_img) @ emb_img.t()
            score3_sp = (rel_img * arg1_real) @ emb_img.t()
            score4_sp = (rel_img * arg1_img) @ emb_real.t()

            score_sp = score1_sp + score2_sp + score3_sp - score4_sp

        if arg2 is not None:
            arg2_real, arg2_img = arg2[:, :self.embedding_size], arg2[:, self.embedding_size:]

            score1_po = (rel_real * arg2_real) @ emb_real.t()
            score2_po = (rel_real * arg2_img) @ emb_img.t()
            score3_po = (rel_img * arg2_img) @ emb_real.t()
            score4_po = (rel_img * arg2_real) @ emb_img.t()

            score_po = score1_po + score2_po + score3_po - score4_po

        embeddings = {
            's': arg1,
            'r': rel,
            'o': arg2
        }

        return score_sp, score_po, embeddings

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        vec_real = embedding_vector[:, :self.embedding_size]
        vec_img = embedding_vector[:, self.embedding_size:]
        return torch.sqrt(vec_real ** 2 + vec_img ** 2)
