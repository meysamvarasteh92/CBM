"""
Build per-ITEM concept vectors and save to disk.
Run ONCE before training SASRec_CBM.

The cache stores a [n_items, n_concepts] matrix where each row is an item's
"intrinsic" concept vector (one-hot for tier / era, multi-hot for genres).

At training/eval time, use `seq_to_concepts(seq, item_concepts, ...)` to
aggregate item-level vectors into a sequence-level concept vector.
"""
import pickle
import numpy as np
import pandas as pd
import torch


def build_lookups(dataset, dataset_name):
    n_items = dataset.item_num
    iid_field = dataset.iid_field

    # ── Tier lookups (already using internal IDs via inter_feat) ─────────────
    inter_df = pd.DataFrame({
        'item_id': dataset.inter_feat[iid_field].numpy(),
    })
    item_counts = inter_df.groupby('item_id').size()
    counts_arr = np.array([item_counts.get(i, 0) for i in range(n_items)])

    niche_threshold = np.percentile(counts_arr, 10)
    pop_threshold   = np.percentile(counts_arr, 90)

    pop_set   = {i for i in range(n_items) if counts_arr[i] >= pop_threshold}
    mid_set   = {i for i in range(n_items)
                 if niche_threshold < counts_arr[i] < pop_threshold}
    niche_set = {i for i in range(n_items)
                 if 0 < counts_arr[i] <= niche_threshold}

    print(f"  pop_set:   {len(pop_set)} items (≥ {pop_threshold:.0f} interactions)")
    print(f"  mid_set:   {len(mid_set)} items")
    print(f"  niche_set: {len(niche_set)} items (1–{niche_threshold:.0f} interactions)")

    # ── Load item metadata ───────────────────────────────────────────────────
    item_path = f'./dataset/{dataset_name}/{dataset_name}.item'
    items = pd.read_csv(item_path, sep='\t')

    def col(prefix):
        for c in items.columns:
            if c.startswith(prefix):
                return c
        return None

    iid_c   = col('item_id')
    genre_c = col('genre')
    year_c  = col('release_year') or col('year')
    if iid_c is None or genre_c is None or year_c is None:
        raise ValueError(f"Couldn't find expected columns in {item_path}. "
                         f"Got: {items.columns.tolist()}")

    # ── CRITICAL: map raw item IDs → RecBole internal IDs ────────────────────
    # dataset.field2token_id[iid_field] is {raw_id_str: internal_id_int}
    item_id_map = dataset.field2token_id[iid_field]
    items['internal_id'] = items[iid_c].astype(str).map(item_id_map)

    n_before = len(items)
    items = items.dropna(subset=['internal_id'])           # drop unmatched
    items['internal_id'] = items['internal_id'].astype(int)
    n_after = len(items)
    print(f"  Items in metadata file: {n_before}, matched to internal IDs: {n_after}")
    if n_after == 0:
        raise ValueError(
            "No items in metadata file matched to RecBole internal IDs. "
            "Check that dataset.field2token_id[iid_field] is populated."
        )

    # ── Genre list (uses space split, as ml-1mm format requires) ─────────────
    all_genres = sorted(set(
        g.strip()
        for genres in items[genre_c].dropna()
        for g in str(genres).split(' ')
        if g.strip()
    ))
    genre_concepts = sorted(all_genres)

    # ── item_to_genres keyed by internal_id ──────────────────────────────────
    item_to_genres = {
        int(r['internal_id']): set(str(r[genre_c]).split(' '))
        for _, r in items.iterrows()
        if pd.notna(r[genre_c])
    }

    # ── Era assignment, keyed by internal_id ─────────────────────────────────
    items['year_num'] = pd.to_numeric(items[year_c], errors='coerce')

    item_to_era = {}
    for _, r in items.iterrows():
        y = r['year_num']
        if pd.isna(y):              e = None
        elif y < 1970:              e = 'classic'
        elif 1970 <= y < 1990:      e = 'retro'
        elif 1990 <= y < 2000:      e = 'modern'
        elif 2000 <= y:             e = 'contemporary'
        item_to_era[int(r['internal_id'])] = e

    era_idx = {'classic': 0, 'retro': 1, 'modern': 2, 'contemporary': 3}

    scalar_concepts = ['popularity', 'mid', 'niche',
                       'classic', 'retro', 'modern', 'contemporary']
    concept_names = genre_concepts + scalar_concepts
    N_CONCEPTS = len(concept_names)

    print(f"  N_CONCEPTS = {N_CONCEPTS}  "
          f"({len(genre_concepts)} genres + {len(scalar_concepts)} scalars)")
    print(f"  genres: {genre_concepts}")

    return dict(
        genre_concepts=genre_concepts,
        item_to_genres=item_to_genres,
        pop_set=pop_set, mid_set=mid_set, niche_set=niche_set,
        item_to_era=item_to_era, era_idx=era_idx,
        concept_names=concept_names,
        N_CONCEPTS=N_CONCEPTS,
        n_items=n_items,
    )


def compute_item_concepts(item_id, L):
    """
    Return the concept vector for a single item (queried by INTERNAL id).
    Layout:
      [0:ng]      multi-hot genres (1.0 for each genre this item has)
      [ng]        1.0 if popular, else 0
      [ng+1]      1.0 if mid,     else 0
      [ng+2]      1.0 if niche,   else 0
      [ng+3:..+7] one-hot era (classic / retro / modern / contemporary)
    """
    vec = np.zeros(L['N_CONCEPTS'], dtype=np.float32)
    ng = len(L['genre_concepts'])

    if item_id == 0:                         # padding
        return vec

    # Genres
    gi = {g: i for i, g in enumerate(L['genre_concepts'])}
    for g in L['item_to_genres'].get(item_id, ()):
        if g in gi:
            vec[gi[g]] = 1.0
    
    # Popularity tier (mutually exclusive)
    if item_id in L['pop_set']:
        vec[ng + 0] = 1.0
    elif item_id in L['mid_set']:
        vec[ng + 1] = 1.0
    elif item_id in L['niche_set']:
        vec[ng + 2] = 1.0

    # Era (mutually exclusive)
    era = L['item_to_era'].get(item_id)
    if era in L['era_idx']:
        vec[ng + 3 + L['era_idx'][era]] = 1.0

    return vec


def seq_to_concepts(item_seq_batch, item_concepts, n_concepts, n_genres,
                    n_tier=3, n_era=4):
    """
    Aggregate item-level concept vectors into sequence-level concept vectors,
    for a batch of sequences.

    Args:
        item_seq_batch: [B, max_seq_len] array or tensor of item IDs (padding = 0)
        item_concepts:  [n_items, n_concepts] item concept matrix
        n_concepts:     total number of concepts
        n_genres:       number of genre slots at the start
        n_tier:         number of tier slots (default 3: pop/mid/niche)
        n_era:          number of era slots (default 4)

    Returns:
        [B, n_concepts] array — same dtype as item_concepts
    """
    is_tensor = isinstance(item_seq_batch, torch.Tensor)
    if is_tensor:
        device = item_seq_batch.device
        if not isinstance(item_concepts, torch.Tensor):
            item_concepts = torch.from_numpy(item_concepts).to(device)
        elif item_concepts.device != device:
            item_concepts = item_concepts.to(device)

        B, S = item_seq_batch.shape
        flat = item_seq_batch.reshape(-1)
        flat_vecs = item_concepts[flat]
        vecs = flat_vecs.view(B, S, -1)
        mask = (item_seq_batch != 0).float().unsqueeze(-1)
        summed = (vecs * mask).sum(dim=1)
        n = mask.sum(dim=1).clamp(min=1)

        out = torch.zeros_like(summed)
        out[:, :n_genres]                  = summed[:, :n_genres] / n
        out[:, n_genres:n_genres+n_tier]   = summed[:, n_genres:n_genres+n_tier] / n

        era_off = n_genres + n_tier
        era_counts = summed[:, era_off:era_off+n_era]
        n_known = era_counts.sum(dim=1, keepdim=True).clamp(min=1)
        out[:, era_off:era_off+n_era] = era_counts / n_known

        return out

    else:
        item_seq_batch = np.asarray(item_seq_batch)
        B, S = item_seq_batch.shape

        flat = item_seq_batch.reshape(-1)
        flat_vecs = item_concepts[flat]
        vecs = flat_vecs.reshape(B, S, -1)

        mask = (item_seq_batch != 0).astype(np.float32)[..., None]
        summed = (vecs * mask).sum(axis=1)
        n = mask.sum(axis=1)
        n = np.maximum(n, 1)

        out = np.zeros_like(summed)
        out[:, :n_genres]                  = summed[:, :n_genres] / n
        out[:, n_genres:n_genres+n_tier]   = summed[:, n_genres:n_genres+n_tier] / n

        era_off = n_genres + n_tier
        era_counts = summed[:, era_off:era_off+n_era]
        n_known = np.maximum(era_counts.sum(axis=1, keepdims=True), 1.0)
        out[:, era_off:era_off+n_era] = era_counts / n_known

        return out


# ─────────────────────────────────────────────────────────────────────────────
# BUILD — main entry
# ─────────────────────────────────────────────────────────────────────────────
def build_cache(dataset_name, dataset):
    print("[build] Building concept lookups...")
    L = build_lookups(dataset, dataset_name)

    print(f"[build] Computing per-item concept vectors for {L['n_items']} items...")
    item_concepts = np.zeros((L['n_items'], L['N_CONCEPTS']), dtype=np.float32)
    for item_id in range(L['n_items']):
        item_concepts[item_id] = compute_item_concepts(item_id, L)

    out_path = f"./dataset/{dataset_name}/saved_concept_individual_items.pkl"
    # Build the cache dict
    cache_data = {
        'item_concepts': item_concepts,
        'concept_names': L['concept_names'],
        'n_concepts':    L['N_CONCEPTS'],
        'n_genres':      len(L['genre_concepts']),
        'n_items':       L['n_items'],
        }

    # Save to disk
    with open(out_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"\n[build] Saved per-item concept matrix ({item_concepts.shape}) → {out_path}")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    n_genres = len(L['genre_concepts'])
    genre_sums = item_concepts[:, :n_genres].sum(axis=1)
    era_sums   = item_concepts[:, n_genres+3:n_genres+7].sum(axis=1)
    tier_sums  = item_concepts[:, n_genres:n_genres+3].sum(axis=1)

    print(f"\n[build] Diagnostics — items with NO ...")
    print(f"    ... genre features:  {(genre_sums == 0).sum()} (should be ~1, padding)")
    print(f"    ... era features:    {(era_sums == 0).sum()} (only items missing year)")
    print(f"    ... tier features:   {(tier_sums == 0).sum()} (only items w/ 0 interactions)")
    print(f"    mean genres/item:    {genre_sums[1:].mean():.2f}")    # skip padding

    print(f"\n[build] Spot check — item 1's concept vector:")

    for name, val in zip(L['concept_names'], item_concepts[1]):
        if val > 0:
            print(f"    {name:<20s} {val:.4f}")

    return cache_data, L['concept_names']