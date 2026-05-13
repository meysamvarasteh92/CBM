import argparse
import logging
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.sasrec_cbm import SASRec_CBM
from recbole.evaluator import Collector, Evaluator
from recbole.quick_start import load_data_and_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         type=str,   default='ml-1mm')
    parser.add_argument('--config',          type=str,   default='CBM_config.yaml')
    parser.add_argument('--lambda_concept',  type=float, default=1.0)
    parser.add_argument('--lambda_recon',    type=float, default=1.0)
    parser.add_argument('--epochs',          type=int,   default=None)
    parser.add_argument('--skip_baseline',   action='store_true',
                        help="Skip the SASRec baseline evaluation step.")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    config_dict = {
        'lambda_concept': args.lambda_concept,
        'lambda_recon':   args.lambda_recon,
    }
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs

    config = Config(
        model='SASRec_CBM',
        dataset=args.dataset,
        config_file_list=[args.config],
        config_dict=config_dict,
    )

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # ── Step 1: Baseline SASRec evaluation ──────────────────────────────────
    sasrec_test_result = None
    if not args.skip_baseline:
        logger.info("=" * 70)
        logger.info("Step 1 — Evaluating frozen SASRec baseline")
        logger.info("=" * 70)
        sasrec_test_result = evaluate_sasrec_baseline(config, train_data, test_data)
        print("sasrec_test_result:",sasrec_test_result)
        logger.info(f"SASRec baseline test result: {sasrec_test_result}")

    # ── Step 2: Train CBM ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Step 2 — Training CBM "
                f"(lambda_concept={args.lambda_concept}, "
                f"lambda_recon={args.lambda_recon})")
    logger.info("=" * 70)

    model = SASRec_CBM(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)
    trainer.eval_collector.data_collect(train_data)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data,
        saved=True,
        show_progress=True,
    )
    logger.info(f"Best valid score: {best_valid_score:.4f}")
    logger.info(f"Best valid result: {best_valid_result}")

    # ── Step 3: Test CBM ─────────────────────────────────────────────────────
    cbm_test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
    logger.info(f"CBM test result: {cbm_test_result}")

    # ── Step 4: Side-by-side comparison ──────────────────────────────────────
    if sasrec_test_result is not None:
        logger.info("=" * 70)
        logger.info("Comparison: SASRec baseline vs CBM")
        logger.info("=" * 70)
        print_comparison(sasrec_test_result, cbm_test_result)

    return best_valid_result, cbm_test_result, sasrec_test_result


def evaluate_sasrec_baseline(config, train_data, test_data):
    """Evaluate the pretrained SASRec checkpoint that the CBM uses as base_path."""
    base_path = config['base_path']

    logger = logging.getLogger()
    logger.info(f"Loading SASRec from {base_path}")
    
    # Save original metrics, swap in non-CBM metrics for the baseline eval
    original_metrics = config['metrics']
    config['metrics'] = [m for m in original_metrics
                         if m not in ('ConceptAccuracySoft', 'ConceptAccuracyStrict')]
    
    # Build a vanilla SASRec model
    sasrec = SASRec(config, train_data.dataset).to(config['device'])

    # Load the pretrained weights
    ckpt = torch.load(base_path, map_location=config['device'], weights_only=False)
    sasrec.load_state_dict(ckpt['state_dict'])
    sasrec.eval()

    # Rebuild trainer with the modified metrics
    trainer = Trainer(config, sasrec)
    trainer.eval_collector.data_collect(train_data)
    trainer.eval_collector = Collector(config)

    trainer.evaluator      = Evaluator(config)
    trainer.eval_collector.data_collect(train_data)

    try:
        result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    finally:
        # Restore original metrics for the CBM run
        config['metrics'] = original_metrics
    
    

    return result

def print_comparison(sasrec_result, cbm_result):
    """Side-by-side print of SASRec vs CBM metrics."""
    print(f"\n{'Metric':<25} {'SASRec':>12} {'CBM':>12} {'Δ (CBM-SASRec)':>20}")
    print("─" * 72)
    for key in sasrec_result.keys():
        s = sasrec_result.get(key, float('nan'))
        c = cbm_result.get(key,    float('nan'))
        print(f"{key:<25} {s:>12.4f} {c:>12.4f} {c-s:>+20.4f}")


if __name__ == '__main__':
    main()