import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.sasrec import SASRec


class TiedSparseAutoencoder(nn.Module):
    """
    Sparse autoencoder used on both:
      1) SASRec sequence/user representations h
      2) SASRec item embeddings e_i

    Encoder:
        z = ReLU(W h + b_enc)

    Tied decoder:
        h_hat = W^T z + b_dec
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder_bias)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, z):
        # encoder.weight: [latent_dim, input_dim]
        # decoder weight: [input_dim, latent_dim]
        return F.linear(z, self.encoder.weight.t(), self.decoder_bias)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class SASRec_Mon_SAE(SASRec):
    """
    Post-hoc interaction-aware Sparse Autoencoder baseline on top of
    a frozen pretrained SASRec.

    SASRec:
        item_seq -> frozen SASRec -> h

    Shared SAE:
        h   -> z_u -> h_hat
        e_i -> z_i -> e_i_hat

    Original SASRec score:
        s(u, i) = h^T e_i

    Reconstructed score:
        s_hat(u, i) = h_hat^T e_i_hat

    Loss:
        alpha * L_embedding
      + beta  * L_prediction
      + lambda_l1 * L1
      + lambda_kl * KL

    The base SASRec is frozen. Only the SAE is trained.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = config["hidden_size"]

        # ------------------------------------------------------------
        # Load pretrained SASRec
        # ------------------------------------------------------------
        base_path = config["base_path"]
        ckpt = torch.load(
            base_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)

        # ------------------------------------------------------------
        # Freeze ALL inherited SASRec parameters first.
        # The SAE is created afterward, so it remains trainable.
        # ------------------------------------------------------------
        for p in self.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------
        # SAE hyperparameters
        # ------------------------------------------------------------
        cfg = config.final_config_dict

        self.sae_dim = cfg.get("sae_dim", 22)

        self.lambda_emb = cfg.get("lambda_emb", 1.0)
        self.lambda_pred = cfg.get("lambda_pred", 1.0)
        self.lambda_l1 = cfg.get("lambda_l1", 1e-4)
        self.lambda_kl = cfg.get("lambda_kl", 1e-2)

        # Desired average activation for each SAE neuron.
        self.sae_rho = cfg.get("sae_rho", 0.05)

        # Number of item embeddings sampled per training batch for
        # the prediction-aware loss. Positive targets are also added.
        #
        # Set to 0 to use the entire item catalog.
        self.sae_num_sampled_items = cfg.get(
            "sae_num_sampled_items", 256
        )

        # Optional steering at evaluation time.
        self.steer_neuron_idx = cfg.get("steer_neuron_idx", None)
        self.steer_scale = cfg.get("steer_scale", 1.0)
        self.steer_add = cfg.get("steer_add", 0.0)

        # ------------------------------------------------------------
        # Shared SAE for sequence and item representations
        # ------------------------------------------------------------
        self.sae = TiedSparseAutoencoder(
            input_dim=self.hidden_size,
            latent_dim=self.sae_dim,
        )

        self._last_losses = {}

        # Ensure frozen SASRec components stay in evaluation mode.
        self._set_backbone_eval()

    # ==============================================================
    # Frozen SASRec encoder
    # ==============================================================

    def _set_backbone_eval(self):
        """
        Keep the pretrained SASRec backbone deterministic while the
        SAE is being trained.
        """
        for module_name in [
            "position_embedding",
            "item_embedding",
            "LayerNorm",
            "trm_encoder",
            "dropout",
        ]:
            if hasattr(self, module_name):
                getattr(self, module_name).eval()

    def _encode(self, item_seq, item_seq_len):
        """
        Returns the exact final SASRec representation used for
        next-item scoring.
        """
        with torch.no_grad():
            position_ids = torch.arange(
                item_seq.size(1),
                dtype=torch.long,
                device=item_seq.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

            position_embedding = self.position_embedding(position_ids)
            item_emb = self.item_embedding(item_seq)

            input_emb = self.LayerNorm(item_emb + position_embedding)

            # Keep this identical to your current SASRec/CBM code.
            # If your installed RecBole SASRec includes an explicit
            # input dropout here, add:
            # input_emb = self.dropout(input_emb)

            extended_attention_mask = self.get_attention_mask(item_seq)

            trm_output = self.trm_encoder(
                input_emb,
                extended_attention_mask,
                output_all_encoded_layers=True,
            )

            output = trm_output[-1]
            h = self.gather_indexes(
                output,
                item_seq_len - 1,
            )

        return h

    # ==============================================================
    # SAE utilities
    # ==============================================================

    def _kl_sparsity(self, z, eps=1e-6):
        """
        KL sparsity penalty.

        z comes from ReLU and is nonnegative. We use the batch mean
        activation as the empirical activation level and clamp it to
        (0, 1) before applying the Bernoulli KL penalty from the paper.

        If you later copy the authors' exact KL helper from their
        repository, you can replace only this function.
        """
        rho = torch.tensor(
            self.sae_rho,
            dtype=z.dtype,
            device=z.device,
        )

        rho_hat = z.mean(dim=0)
        rho_hat = torch.clamp(rho_hat, eps, 1.0 - eps)

        kl = (
            rho * torch.log((rho + eps) / rho_hat)
            + (1.0 - rho)
            * torch.log(
                (1.0 - rho + eps) /
                (1.0 - rho_hat)
            )
        )

        return kl.mean()

    def _sample_items(self, interaction):
        """
        Sample a shared candidate set for the prediction-aware loss.

        We also include the positive next items in the current batch.
        """
        # RecBole sequential models normally expose POS_ITEM_ID.
        pos_items = interaction[self.POS_ITEM_ID].view(-1)

        if self.sae_num_sampled_items <= 0:
            candidate_ids = torch.arange(
                1,
                self.n_items,
                device=pos_items.device,
                dtype=torch.long,
            )
        else:
            random_ids = torch.randint(
                low=1,
                high=self.n_items,
                size=(self.sae_num_sampled_items,),
                device=pos_items.device,
            )

            candidate_ids = torch.cat(
                [pos_items, random_ids],
                dim=0,
            )

            candidate_ids = torch.unique(candidate_ids)

        return candidate_ids

    def _reconstruct_items(self, item_ids):
        """
        Reconstruct selected SASRec item embeddings with the same SAE.
        """
        with torch.no_grad():
            item_emb = self.item_embedding(item_ids)

        item_hat, z_item = self.sae(item_emb)

        return item_emb, z_item, item_hat

    def _reconstruct_all_items(self):
        """
        Reconstruct the whole SASRec output embedding table.

        Used for full-sort evaluation.
        """
        with torch.no_grad():
            item_embs = self.item_embedding.weight.detach()

        item_hat, z_item = self.sae(item_embs)

        # Item 0 is RecBole's padding item. Keep its decoded vector at 0.
        item_hat = item_hat.clone()
        item_hat[0] = 0.0

        return item_embs, z_item, item_hat

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(self, item_seq, item_seq_len):
        """
        Returns:
            h      : frozen SASRec sequence representation [B, d]
            z_user : sparse SAE representation               [B, m]
            h_hat  : reconstructed SASRec representation      [B, d]
        """
        h = self._encode(item_seq, item_seq_len)

        h_hat, z_user = self.sae(h)

        # Optional neuron intervention for later steering experiments.
        if self.steer_neuron_idx is not None and not self.training:
            z_user = z_user.clone()

            idx = int(self.steer_neuron_idx)

            if self.steer_scale != 1.0:
                z_user[:, idx] *= self.steer_scale

            if self.steer_add != 0.0:
                z_user[:, idx] += self.steer_add

            h_hat = self.sae.decode(z_user)

        return h, z_user, h_hat

    # ==============================================================
    # Training loss
    # ==============================================================

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # ----------------------------------------------------------
        # 1. Frozen SASRec sequence representation
        # ----------------------------------------------------------
        h, z_user, h_hat = self.forward(
            item_seq,
            item_seq_len,
        )

        # ----------------------------------------------------------
        # 2. Sample item representations
        # ----------------------------------------------------------
        candidate_ids = self._sample_items(interaction)

        item_emb, z_item, item_hat = self._reconstruct_items(
            candidate_ids
        )

        # ----------------------------------------------------------
        # 3. Embedding-level reconstruction loss
        # ----------------------------------------------------------
        loss_user_recon = F.mse_loss(
            h_hat,
            h.detach(),
        )

        loss_item_recon = F.mse_loss(
            item_hat,
            item_emb.detach(),
        )

        # Balance the two representation types equally.
        loss_emb = 0.5 * (
            loss_user_recon + loss_item_recon
        )

        # ----------------------------------------------------------
        # 4. Prediction-aware reconstruction loss
        #
        # Original SASRec:
        #     scores = h @ e^T
        #
        # Reconstructed:
        #     scores_hat = h_hat @ e_hat^T
        # ----------------------------------------------------------
        with torch.no_grad():
            original_scores = (
                h.detach() @ item_emb.detach().t()
            )

        reconstructed_scores = (
            h_hat @ item_hat.t()
        )

        loss_pred = F.mse_loss(
            reconstructed_scores,
            original_scores,
        )

        # ----------------------------------------------------------
        # 5. Sparsity
        # ----------------------------------------------------------
        loss_l1_user = z_user.abs().mean()
        loss_l1_item = z_item.abs().mean()

        loss_l1 = 0.5 * (
            loss_l1_user + loss_l1_item
        )

        loss_kl_user = self._kl_sparsity(z_user)
        loss_kl_item = self._kl_sparsity(z_item)

        loss_kl = 0.5 * (
            loss_kl_user + loss_kl_item
        )

        # ----------------------------------------------------------
        # 6. Total SAE objective
        # ----------------------------------------------------------
        loss = (
            self.lambda_emb * loss_emb
            + self.lambda_pred * loss_pred
            + self.lambda_l1 * loss_l1
            + self.lambda_kl * loss_kl
        )

        self._last_losses = {
            "total": float(loss.detach().cpu()),
            "emb": float(loss_emb.detach().cpu()),
            "user_recon": float(loss_user_recon.detach().cpu()),
            "item_recon": float(loss_item_recon.detach().cpu()),
            "pred": float(loss_pred.detach().cpu()),
            "l1": float(loss_l1.detach().cpu()),
            "kl": float(loss_kl.detach().cpu()),
            "mean_user_activation": float(
                z_user.detach().mean().cpu()
            ),
            "mean_item_activation": float(
                z_item.detach().mean().cpu()
            ),
            "user_zero_fraction": float(
                (z_user.detach() <= 0).float().mean().cpu()
            ),
            "item_zero_fraction": float(
                (z_item.detach() <= 0).float().mean().cpu()
            ),
        }

        return loss

    # ==============================================================
    # RecBole prediction interfaces
    # ==============================================================

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        _, _, h_hat = self.forward(
            item_seq,
            item_seq_len,
        )

        with torch.no_grad():
            test_item_emb = self.item_embedding(test_item)

        test_item_hat, _ = self.sae(test_item_emb)

        scores = torch.mul(
            h_hat,
            test_item_hat,
        ).sum(dim=-1)

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        _, _, h_hat = self.forward(
            item_seq,
            item_seq_len,
        )

        _, _, all_item_hat = self._reconstruct_all_items()

        scores = h_hat @ all_item_hat.t()

        return scores

    # ==============================================================
    # Useful baseline/fidelity helpers
    # ==============================================================

    def full_sort_predict_original(self, interaction):
        """
        Original frozen SASRec ranking.

        Useful for RBO / Kendall-Tau / fidelity comparisons.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        h = self._encode(
            item_seq,
            item_seq_len,
        )

        with torch.no_grad():
            item_embs = self.item_embedding.weight.detach()

        return h @ item_embs.t()

    def get_user_sae_activations(self, interaction):
        """
        Return z_u for neuron interpretation.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        _, z_user, _ = self.forward(
            item_seq,
            item_seq_len,
        )

        return z_user

    def get_item_sae_activations(self):
        """
        Return SAE activations for all catalog items.

        Shape:
            [n_items, sae_dim]

        This is what you will use to identify genre/popularity/
        temporal neurons using the metadata.
        """
        with torch.no_grad():
            item_embs = self.item_embedding.weight.detach()
            z_item = self.sae.encode(item_embs)

        return z_item

    # ==============================================================
    # Make sure frozen SASRec stays in eval mode during SAE training
    # ==============================================================

    def train(self, mode=True):
        super().train(mode)

        if mode:
            self._set_backbone_eval()

        return self
