

'''

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
        #vec[ng + 0 + L['era_idx'][era]] = 1.0
    


  

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
    #era_sums   = item_concepts[:, n_genres:n_genres+4].sum(axis=1)

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

'''

"""
Build per-ITEM concept vectors and save to disk.
Run ONCE before training SASRec_CBM.

Supports multiple datasets via DATASET_CONFIGS:
  - ml-1m:       multi-hot genres + era (classic/retro/modern/contemporary)
  - beeradvocate: one-hot style concept + ABV tier (low/medium/high)

To add a new dataset, add an entry to DATASET_CONFIGS.
"""
import pickle
import numpy as np
import pandas as pd
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configurations
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    'ml-1mm': {
        'concept_type': 'genre_era',
        'item_col':     'item_id',
        'genre_col':    'genre',
        'year_col':     'release_year',
        'genre_sep':    ' ',                        # space-separated genres
        'era_bins':     [1970, 1990, 2000],
        'era_labels':   ['classic', 'retro', 'modern', 'contemporary'],
    },
    'beeradvocate': {
    'concept_type':    'style_abv',
    'item_col':        'item_id',
    'style_col':       'style_concept',  # matches exactly
    'abv_col':         'beer/ABV',       # fix this
    'style_concepts':  [
        'IPA', 'Stout', 'Porter', 'Pale Ale', 'Lager', 'Dark Lager',
        'Wheat', 'Belgian', 'Sour', 'Bock', 'Amber Ale',
        'Bitter', 'Strong Ale', 'Specialty', 'Low Alcohol',
    ],
    'abv_bins':   [4.5, 7.0, 15],
    'abv_labels': ['abv_low', 'abv_medium', 'abv_high', 'abv_extreme'],
},
   
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col(df, prefix):
    """Find first column starting with prefix (handles :token/:float suffixes)."""
    for c in df.columns:
        if c.startswith(prefix):
            return c
    return None


def _assign_era(year, bins, labels):
    """Assign era label based on year and bin edges."""
    if pd.isna(year):
        return None
    thresholds = bins
    for i, t in enumerate(thresholds):
        if year < t:
            return labels[i]
    return labels[-1]


def _assign_abv_bin(abv, bins, labels):
    """Assign ABV bin label."""
    if pd.isna(abv):
        return None
    for i, t in enumerate(bins):
        if abv < t:
            return labels[i]
    return labels[-1]


# ─────────────────────────────────────────────────────────────────────────────
# build_lookups — dataset-aware
# ─────────────────────────────────────────────────────────────────────────────

def build_lookups(dataset, dataset_name):
    cfg = DATASET_CONFIGS.get(dataset_name)
    if cfg is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in DATASET_CONFIGS. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    n_items   = dataset.item_num
    iid_field = dataset.iid_field

    # ── Popularity tiers ─────────────────────────────────────────────────────
    inter_df = pd.DataFrame({'item_id': dataset.inter_feat[iid_field].numpy()})
    item_counts   = inter_df.groupby('item_id').size()
    counts_arr    = np.array([item_counts.get(i, 0) for i in range(n_items)])

    niche_threshold = np.percentile(counts_arr, 10)
    pop_threshold   = np.percentile(counts_arr, 90)

    pop_set   = {i for i in range(n_items) if counts_arr[i] >= pop_threshold}
    mid_set   = {i for i in range(n_items)
                 if niche_threshold < counts_arr[i] < pop_threshold}
    niche_set = {i for i in range(n_items)
                 if 0 < counts_arr[i] <= niche_threshold}

    print(f"  pop_set:   {len(pop_set):,} items (≥ {pop_threshold:.0f} interactions)")
    print(f"  mid_set:   {len(mid_set):,} items")
    print(f"  niche_set: {len(niche_set):,} items (1–{niche_threshold:.0f} interactions)")

    # ── Load item metadata ───────────────────────────────────────────────────
    item_path = f'./dataset/{dataset_name}/{dataset_name}.item'
    items = pd.read_csv(item_path, sep='\t')

    iid_c = _col(items, cfg['item_col'])
    if iid_c is None:
        raise ValueError(f"item_col '{cfg['item_col']}' not found. Got: {items.columns.tolist()}")

    # Map raw item IDs → RecBole internal IDs
    item_id_map = dataset.field2token_id[iid_field]
    items['internal_id'] = items[iid_c].astype(str).map(item_id_map)
    

    n_before = len(items)
    items = items.dropna(subset=['internal_id'])
    items['internal_id'] = items['internal_id'].astype(int)
    n_after = len(items)
    print(f"  Items in metadata: {n_before:,}, matched to internal IDs: {n_after:,}")
    if n_after == 0:
        raise ValueError(
            "No items matched to RecBole internal IDs. "
            "Check that dataset.field2token_id[iid_field] is populated."
        )

    # ── Build concept-type-specific lookups ──────────────────────────────────
    concept_type = cfg['concept_type']

    if concept_type == 'genre_era':
        return _build_genre_era(items, cfg, pop_set, mid_set, niche_set, n_items)

    elif concept_type == 'style_abv':
        return _build_style_abv(items, cfg, pop_set, mid_set, niche_set, n_items)

    else:
        raise ValueError(f"Unknown concept_type: '{concept_type}'")


# ── genre_era (ML-1M) ────────────────────────────────────────────────────────

def _build_genre_era(items, cfg, pop_set, mid_set, niche_set, n_items):
    genre_c = _col(items, cfg['genre_col'])
    year_c  = _col(items, cfg['year_col'])
    sep     = cfg.get('genre_sep', ' ')

    if genre_c is None or year_c is None:
        raise ValueError(
            f"Could not find genre_col='{cfg['genre_col']}' or "
            f"year_col='{cfg['year_col']}'. Got: {items.columns.tolist()}"
        )

    # Genre vocab
    all_genres = sorted(set(
        g.strip()
        for gs in items[genre_c].dropna()
        for g in str(gs).split(sep)
        if g.strip()
    ))
    genre_concepts = sorted(all_genres)

    item_to_genres = {
        int(r['internal_id']): set(str(r[genre_c]).split(sep))
        for _, r in items.iterrows()
        if pd.notna(r[genre_c])
    }

    # Era assignment
    items['year_num'] = pd.to_numeric(items[year_c], errors='coerce')
    era_labels = cfg['era_labels']
    era_bins   = cfg['era_bins']
    era_idx    = {e: i for i, e in enumerate(era_labels)}

    item_to_era = {
        int(r['internal_id']): _assign_era(r['year_num'], era_bins, era_labels)
        for _, r in items.iterrows()
    }

    scalar_concepts = ['popularity', 'mid', 'niche'] + era_labels
    concept_names   = genre_concepts + scalar_concepts
    N_CONCEPTS      = len(concept_names)

    print(f"  N_CONCEPTS = {N_CONCEPTS}  "
          f"({len(genre_concepts)} genres + {len(scalar_concepts)} scalars)")

    return dict(
        concept_type='genre_era',
        genre_concepts=genre_concepts,
        item_to_genres=item_to_genres,
        pop_set=pop_set, mid_set=mid_set, niche_set=niche_set,
        item_to_era=item_to_era, era_idx=era_idx,
        n_era=len(era_labels),
        concept_names=concept_names,
        N_CONCEPTS=N_CONCEPTS,
        n_items=n_items,
        n_primary=len(genre_concepts),      # unified key: primary concept count
    )


# ── style_abv (BeerAdvocate) ──────────────────────────────────────────────────

def _build_style_abv(items, cfg, pop_set, mid_set, niche_set, n_items):
    style_c = _col(items, cfg['style_col'])
    abv_c   = _col(items, cfg['abv_col'])

    if style_c is None or abv_c is None:
        raise ValueError(
            f"Could not find style_col='{cfg['style_col']}' or "
            f"abv_col='{cfg['abv_col']}'. Got: {items.columns.tolist()}"
        )

    style_concepts = cfg['style_concepts']
    abv_labels     = cfg['abv_labels']
    abv_bins       = cfg['abv_bins']
    style_idx      = {s: i for i, s in enumerate(style_concepts)}
    abv_idx        = {b: i for i, b in enumerate(abv_labels)}

    item_to_style = {
        int(r['internal_id']): r[style_c]
        for _, r in items.iterrows()
        if pd.notna(r[style_c])
    }

    items['abv_num'] = pd.to_numeric(items[abv_c], errors='coerce')
    item_to_abv = {
        int(r['internal_id']): _assign_abv_bin(r['abv_num'], abv_bins, abv_labels)
        for _, r in items.iterrows()
    }

    scalar_concepts = ['popularity', 'mid', 'niche'] + abv_labels
    concept_names   = style_concepts + scalar_concepts
    N_CONCEPTS      = len(concept_names)

    print(f"  N_CONCEPTS = {N_CONCEPTS}  "
          f"({len(style_concepts)} styles + {len(scalar_concepts)} scalars)")

    return dict(
        concept_type='style_abv',
        style_concepts=style_concepts,
        style_idx=style_idx,
        item_to_style=item_to_style,
        abv_idx=abv_idx,
        item_to_abv=item_to_abv,
        pop_set=pop_set, mid_set=mid_set, niche_set=niche_set,
        n_abv=len(abv_labels),
        concept_names=concept_names,
        N_CONCEPTS=N_CONCEPTS,
        n_items=n_items,
        n_primary=len(style_concepts),      # unified key
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute_item_concepts — dispatches on concept_type
# ─────────────────────────────────────────────────────────────────────────────

def compute_item_concepts(item_id, L):
    if item_id == 0:
        return np.zeros(L['N_CONCEPTS'], dtype=np.float32)

    if L['concept_type'] == 'genre_era':
        return _compute_genre_era(item_id, L)
    elif L['concept_type'] == 'style_abv':
        return _compute_style_abv(item_id, L)
    else:
        raise ValueError(f"Unknown concept_type: {L['concept_type']}")


def _compute_genre_era(item_id, L):
    """
    Layout:
      [0:ng]        multi-hot genres
      [ng]          popular
      [ng+1]        mid
      [ng+2]        niche
      [ng+3:ng+3+ne] one-hot era
    """
    vec = np.zeros(L['N_CONCEPTS'], dtype=np.float32)
    ng  = len(L['genre_concepts'])
    gi  = {g: i for i, g in enumerate(L['genre_concepts'])}

    for g in L['item_to_genres'].get(item_id, ()):
        if g in gi:
            vec[gi[g]] = 1.0

    if item_id in L['pop_set']:   vec[ng + 0] = 1.0
    elif item_id in L['mid_set']: vec[ng + 1] = 1.0
    elif item_id in L['niche_set']:vec[ng + 2] = 1.0

    era = L['item_to_era'].get(item_id)
    if era in L['era_idx']:
        vec[ng + 3 + L['era_idx'][era]] = 1.0

    return vec


def _compute_style_abv(item_id, L):
    """
    Layout:
      [0:ns]        one-hot style concept
      [ns]          popular
      [ns+1]        mid
      [ns+2]        niche
      [ns+3:ns+3+na] one-hot ABV bin
    """
    vec = np.zeros(L['N_CONCEPTS'], dtype=np.float32)
    ns  = len(L['style_concepts'])

    style = L['item_to_style'].get(item_id)
    if style in L['style_idx']:
        vec[L['style_idx'][style]] = 1.0

    if item_id in L['pop_set']:    vec[ns + 0] = 1.0
    elif item_id in L['mid_set']:  vec[ns + 1] = 1.0
    elif item_id in L['niche_set']:vec[ns + 2] = 1.0

    abv_bin = L['item_to_abv'].get(item_id)
    if abv_bin in L['abv_idx']:
        vec[ns + 3 + L['abv_idx'][abv_bin]] = 1.0

    return vec


# ─────────────────────────────────────────────────────────────────────────────
# seq_to_concepts — generic (works for both datasets)
# ─────────────────────────────────────────────────────────────────────────────

def seq_to_concepts(item_seq_batch, item_concepts, n_concepts, n_primary,
                    n_tier=3, n_secondary=None):
    """
    Aggregate item-level concept vectors into sequence-level concept vectors.

    Args:
        item_seq_batch: [B, S] tensor/array of item IDs (0 = padding)
        item_concepts:  [n_items, n_concepts] concept matrix
        n_concepts:     total concept count
        n_primary:      number of primary concept slots (genres or styles)
        n_tier:         number of tier slots (default 3)
        n_secondary:    number of secondary slots (era or ABV bins).
                        If None, inferred as n_concepts - n_primary - n_tier.

    Returns:
        [B, n_concepts] aggregated concept vectors
    """
    if n_secondary is None:
        n_secondary = n_concepts - n_primary - n_tier

    is_tensor = isinstance(item_seq_batch, torch.Tensor)

    if is_tensor:
        device = item_seq_batch.device
        if not isinstance(item_concepts, torch.Tensor):
            item_concepts = torch.from_numpy(item_concepts).to(device)
        elif item_concepts.device != device:
            item_concepts = item_concepts.to(device)

        B, S   = item_seq_batch.shape
        flat   = item_seq_batch.reshape(-1)
        vecs   = item_concepts[flat].view(B, S, -1)
        mask   = (item_seq_batch != 0).float().unsqueeze(-1)
        summed = (vecs * mask).sum(dim=1)
        n      = mask.sum(dim=1).clamp(min=1)

        out = torch.zeros_like(summed)

        # Primary (genres / styles): mean over sequence
        out[:, :n_primary] = summed[:, :n_primary] / n

        # Tier: mean over sequence
        t0 = n_primary
        out[:, t0:t0+n_tier] = summed[:, t0:t0+n_tier] / n

        # Secondary (era / ABV): normalise within known items only
        s0     = t0 + n_tier
        sec    = summed[:, s0:s0+n_secondary]
        n_known = sec.sum(dim=1, keepdim=True).clamp(min=1)
        out[:, s0:s0+n_secondary] = sec / n_known

        return out

    else:
        item_seq_batch = np.asarray(item_seq_batch)
        B, S   = item_seq_batch.shape
        flat   = item_seq_batch.reshape(-1)
        vecs   = item_concepts[flat].reshape(B, S, -1)
        mask   = (item_seq_batch != 0).astype(np.float32)[..., None]
        summed = (vecs * mask).sum(axis=1)
        n      = np.maximum(mask.sum(axis=1), 1)

        out = np.zeros_like(summed)

        out[:, :n_primary] = summed[:, :n_primary] / n

        t0 = n_primary
        out[:, t0:t0+n_tier] = summed[:, t0:t0+n_tier] / n

        s0  = t0 + n_tier
        sec = summed[:, s0:s0+n_secondary]
        n_known = np.maximum(sec.sum(axis=1, keepdims=True), 1.0)
        out[:, s0:s0+n_secondary] = sec / n_known

        return out


# ─────────────────────────────────────────────────────────────────────────────
# build_cache — main entry (same for all datasets)
# ─────────────────────────────────────────────────────────────────────────────

def build_cache(dataset_name, dataset):
    print(f"[build] Dataset: {dataset_name}")
    print("[build] Building concept lookups...")
    L = build_lookups(dataset, dataset_name)

    print(f"[build] Computing per-item concept vectors for {L['n_items']:,} items...")
    item_concepts = np.zeros((L['n_items'], L['N_CONCEPTS']), dtype=np.float32)
    for item_id in range(L['n_items']):
        item_concepts[item_id] = compute_item_concepts(item_id, L)

    out_path = f"./dataset/{dataset_name}/saved_concept_individual_items.pkl"
    cache_data = {
        'item_concepts': item_concepts,
        'concept_names': L['concept_names'],
        'n_concepts':    L['N_CONCEPTS'],
        'n_genres':      L['n_primary'],    # kept for backward compatibility
        'n_primary':     L['n_primary'],
        'n_items':       L['n_items'],
        'concept_type':  L['concept_type'],
    }

    with open(out_path, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"\n[build] Saved concept matrix {item_concepts.shape} → {out_path}")

  

    n_p  = L['n_primary']
    n_t  = 3
    n_s  = L['N_CONCEPTS'] - n_p - n_t

    primary_sums = item_concepts[:, :n_p].sum(axis=1)
    tier_sums    = item_concepts[:, n_p:n_p+n_t].sum(axis=1)
    sec_sums     = item_concepts[:, n_p+n_t:].sum(axis=1)

    print(f"\n[build] Diagnostics — items with NO ...")
    print(f"  ... primary features : {(primary_sums == 0).sum()} (should be ~1, padding)")
    print(f"  ... tier features    : {(tier_sums == 0).sum()} (items w/ 0 interactions)")
    print(f"  ... secondary feats  : {(sec_sums == 0).sum()} (items missing metadata)")
    if L['concept_type'] == 'genre_era':
        print(f"  mean genres/item     : {primary_sums[1:].mean():.2f}")

    print(f"\n[build] Spot check — item 1:")
    for name, val in zip(L['concept_names'], item_concepts[1]):
        if val > 0:
            print(f"  {name:<25s} {val:.4f}")

    return cache_data, L['concept_names']
