import sys
import os
import pytest
import pandas as pd
import torch
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import load_config
from src.model.wrapper import ScGPTWrapper
from src.data.loader import PerturbationDataLoader
from src.utils.metrics import compute_metrics


class TestZeroShotPipeline:
    @pytest.fixture(scope="class")
    def config(self):
        # Load the test config
        config_path = project_root / "tests/integration/config_test.yaml"
        config = load_config(str(config_path))

        # Ensure output directory exists and is clean
        output_dir = Path(config["paths"]["output_dir"])
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Update paths to be absolute to avoid CWD issues
        config["paths"]["data_dir"] = str(project_root / config["paths"]["data_dir"])
        config["paths"]["model_dir"] = str(project_root / config["paths"]["model_dir"])
        config["paths"]["output_dir"] = str(output_dir)

        return config

    @pytest.fixture(scope="class")
    def components(self, config):
        # Initialize core components
        model_wrapper = ScGPTWrapper(config)
        data_loader = PerturbationDataLoader(config, model_wrapper.vocab)
        return model_wrapper, data_loader

    def test_data_loading(self, components):
        """Verify data loader initialization and file reading"""
        _, data_loader = components

        assert data_loader.train_adata is not None
        assert data_loader.test_adata is not None
        assert data_loader.control_cells is not None
        assert data_loader.control_cells.n_obs > 0

        # Check if we can get targets
        targets = data_loader.get_test_targets()
        assert len(targets) > 0

    def test_model_initialization(self, components):
        """Verify model loading and weights"""
        model_wrapper, _ = components

        assert model_wrapper.model is not None
        assert model_wrapper.vocab is not None
        assert len(model_wrapper.vocab) > 0

        # Check if model is in eval mode
        assert not model_wrapper.model.training

    def test_single_prediction_flow(self, components):
        """Run a single prediction and verify output shape and metrics"""
        model_wrapper, data_loader = components

        # Pick a target gene
        targets = data_loader.get_test_targets()
        target = [t for t in targets if t != "non-targeting"][0]

        # Prepare batch
        batch_data = data_loader.prepare_perturbation_batch(target, batch_size=2)
        assert batch_data is not None

        # Predict
        import numpy as np

        gene_ids = np.array(data_loader.test_adata.var["id_in_vocab"])

        with torch.no_grad():
            pred = model_wrapper.predict(batch_data, gene_ids=gene_ids, amp=False)

        # Verify shape
        # Expect (batch_size, n_genes)
        assert pred.shape[0] == 2
        assert pred.shape[1] == data_loader.test_adata.n_vars

        # Verify values are not all zero (sanity check)
        assert torch.any(pred != 0)

    def test_full_pipeline_execution(self, config):
        """Run the main logic equivalent to verify end-to-end flow"""
        # This mimics main.py but just runs for a few iterations
        from src.utils.logging import get_logger

        logger = get_logger("TestPipeline")
        model_wrapper = ScGPTWrapper(config, logger)
        data_loader = PerturbationDataLoader(config, model_wrapper.vocab, logger)

        targets = data_loader.get_test_targets()
        # Limit to 2 targets for speed
        test_targets = [t for t in targets if t != "non-targeting"][:2]

        results = []
        import numpy as np

        for target in test_targets:
            batch_data = data_loader.prepare_perturbation_batch(target)
            if batch_data is None:
                continue

            gene_ids = np.array(data_loader.test_adata.var["id_in_vocab"])
            with torch.no_grad():
                pred = model_wrapper.predict(batch_data, gene_ids=gene_ids, amp=False)

            ground_truth = data_loader.get_target_ground_truth(target)
            if isinstance(ground_truth.X, np.ndarray):
                truth = ground_truth.X
            else:
                truth = ground_truth.X.toarray()

            metrics = compute_metrics(
                truth.mean(axis=0), pred.cpu().numpy().mean(axis=0)
            )
            results.append(metrics)

        assert len(results) == len(test_targets)
        assert "mse" in results[0]
        assert "pearson" in results[0]

        # Save results to verify file writing
        df = pd.DataFrame(results)
        out_path = Path(config["paths"]["output_dir"]) / "test_results.csv"
        df.to_csv(out_path)

        assert out_path.exists()
