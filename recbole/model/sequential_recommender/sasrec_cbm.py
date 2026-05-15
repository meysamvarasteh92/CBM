import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.utils.build_concepts import build_cache, seq_to_concepts

class SASRec_CBM(SASRec):
    """
    Post-hoc Concept Bottleneck Model on top of frozen SASRec.

    Pipeline: item_seq → SASRec encoder → h → concept_predictor → c_hat
                                                       → reconstructor → h_hat
                                                       → score_items
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ── Load pretrained SASRec weights ───────────────────────────────────
        base_path = config["base_path"]
        ckpt = torch.load(base_path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt['state_dict'], strict=False)
        
        self.hidden_size=config['hidden_size']

        self._eval_c_hat_buffer = []
        self._eval_gt_buffer    = []


        # ── Read CBM hyperparameters ─────────────────────────────────────────
       # ── Read CBM hyperparameters ─────────────────────────────────────────
        self.lambda_concept    = config['lambda_concept']    if 'lambda_concept'    in config.final_config_dict else 1.0
        self.lambda_recon      = config['lambda_recon']      if 'lambda_recon'      in config.final_config_dict else 0.5
        
        
        self.steer_concept_idx = config['steer_concept_idx'] if 'steer_concept_idx' in config.final_config_dict else None
        #self.steer_concept_idx2 = config['steer_concept_idx2'] if 'steer_concept_idx2' in config.final_config_dict else None

        self.steer_scale       = config['steer_scale']       if 'steer_scale'       in config.final_config_dict else 1.0
        #self.steer_scale2       = config['steer_scale2']       if 'steer_scale2'       in config.final_config_dict else 1.0

        # ── Load (or build) the per-item concept cache ───────────────────────
        cache_path = f"./dataset/{config['dataset']}/saved_concept_individual_items.pkl"
        #if not os.path.exists(cache_path):
            #print(f"[CBM] Cache not found, building...")
            #build_cache(config['dataset'], dataset)

        #with open(cache_path, 'rb') as f:
            #d = pickle.load(f)
        d, _=build_cache(config['dataset'], dataset)
        # item_concepts: [n_items, n_concepts] — registered as a buffer so it
        # moves with the model (.to(device), state_dict, etc.) but isn't a Parameter
        self.register_buffer(
            'item_concepts',
            torch.from_numpy(d['item_concepts']).float(),
        )
        self.n_concepts = d['n_concepts']
        self.n_genres   = d['n_genres']
        self.concept_names = d['concept_names']

        print(f"[CBM] Loaded {self.item_concepts.shape[0]} items × {self.n_concepts} concepts")

        # ── Trainable modules ────────────────────────────────────────────────
        self.concept_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.n_concepts),
            nn.Sigmoid(),
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(self.n_concepts, 128),
            nn.LayerNorm(128),                 # ← LayerNorm instead of BatchNorm
            nn.GELU(),                         # ← matches SASRec's activation
            nn.Linear(128, self.hidden_size),
        )

        # ── Freeze SASRec by name match, leave CBM trainable ─────────────────
        sasrec_param_names = set(ckpt['state_dict'].keys())
        for name, p in self.named_parameters():
            p.requires_grad = name not in sasrec_param_names

        # SASRec submodules in permanent eval mode (no dropout)
        for module in [self.position_embedding, self.item_embedding,
                       self.LayerNorm, self.trm_encoder, self.dropout]:
            module.eval()

    # ── Concept lookup — vectorized, GPU-resident ────────────────────────────
    def _lookup_concepts(self, item_seq):
        """item_seq: [B, max_seq_len] long tensor → [B, n_concepts] float tensor."""
        return seq_to_concepts(
            item_seq, self.item_concepts,
            self.n_concepts, self.n_genres,
        )

    # ── Encoder ──────────────────────────────────────────────────────────────
    def _encode(self, item_seq, item_seq_len):
      
        with torch.no_grad():
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long,
                                        device=item_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)
            item_emb = self.item_embedding(item_seq)
            input_emb = self.LayerNorm(item_emb + position_embedding)
            #input_emb = self.dropout(input_emb)
            extended_attention_mask = self.get_attention_mask(item_seq)
            trm_output = self.trm_encoder(
                input_emb, extended_attention_mask, output_all_encoded_layers=True
            )
            output = trm_output[-1]
            output = self.gather_indexes(output, item_seq_len - 1)
        return output

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, item_seq, item_seq_len):
        h = self._encode(item_seq, item_seq_len)
        c_hat = self.concept_predictor(h)

        if self.steer_concept_idx is not None and not self.training:
            c_hat = c_hat.clone()
            c_hat[:, self.steer_concept_idx] *= self.steer_scale


        h_hat = self.reconstructor(c_hat)
        return h, c_hat, h_hat

    # ── Training loss ────────────────────────────────────────────────────────
    def calculate_loss(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        target_item  = interaction[self.ITEM_ID]

        # Fast vectorized concept lookup — no Python loop, no CPU round-trip
        gt_concepts = self._lookup_concepts(item_seq)

        h, c_hat, h_hat = self.forward(item_seq, item_seq_len)
        logits = h_hat @ self.item_embedding.weight.T

        #loss_rec   = F.cross_entropy(logits, target_item)
        loss_con   = F.binary_cross_entropy(c_hat, gt_concepts)
        loss_recon = F.mse_loss(h_hat, h.detach())


         # ── Elastic Net on reconstructor weights ─────────────────────────
         # ── Elastic Net (exactly as in the paper) ────────────────────────
        W = self.reconstructor[0].weight          # [128, 25]
        Nc = self.n_concepts                      # 25
        K  = self.hidden_size                     # 128

        elastic_alpha = 0.5                       # balance L1 vs L2
        omega = elastic_alpha * W.abs().sum() + (1 - elastic_alpha) * (W ** 2).sum()

        lambda_reg = 0.01                         # overall strength
        loss_reg = (lambda_reg / (Nc * K)) * omega
        #return loss_rec + self.lambda_concept * loss_con + self.lambda_recon * loss_recon
        return  (self.lambda_concept * loss_con + self.lambda_recon * loss_recon + loss_reg )


    # ── Eval interfaces ──────────────────────────────────────────────────────
    def predict(self, interaction):

        item_seq      = interaction[self.ITEM_SEQ]
        item_seq_len  = interaction[self.ITEM_SEQ_LEN]
        test_item     = interaction[self.ITEM_ID]
        _, c_hat, h_hat   = self.forward(item_seq, item_seq_len)

        if not self.training:
            gt = self._lookup_concepts(item_seq)
            self._eval_c_hat_buffer.append(c_hat.detach().cpu())
            self._eval_gt_buffer.append(gt.detach().cpu())

        test_item_emb = self.item_embedding(test_item)
        return torch.mul(h_hat, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):

        item_seq      = interaction[self.ITEM_SEQ]
        item_seq_len  = interaction[self.ITEM_SEQ_LEN]
        _, c_hat, h_hat   = self.forward(item_seq, item_seq_len)

        self._last_c_hat       = c_hat.detach()
        self._last_gt_concepts = self._lookup_concepts(item_seq).detach()
        if not self.training:
            gt = self._lookup_concepts(item_seq)
            self._eval_c_hat_buffer.append(c_hat.detach().cpu())
            self._eval_gt_buffer.append(gt.detach().cpu())
        return h_hat @ self.item_embedding.weight.T
    
    def train(self, mode=True):
        """Keep frozen SASRec submodules in eval mode even when the parent goes to train."""
        super().train(mode)
        if mode:
            # Force everything inherited from SASRec back to eval
            self.position_embedding.eval()
            self.item_embedding.eval()
            self.LayerNorm.eval()
            self.trm_encoder.eval()       # ← cascades to all internal dropouts
            self.dropout.eval()
        return self