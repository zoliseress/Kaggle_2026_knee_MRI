"""pytest wrappers around the focused failure-mode checks.

    cd src && python -m pytest tests -q          # with pytest installed
    cd src && python -m knee_mri.cli selftest    # same checks, no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knee_mri import selftest  # noqa: E402
from knee_mri.config import load_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return selftest.tiny_config(load_config())


def test_geometric_slice_sorting():
    selftest.check_geometric_sorting()


def test_plane_assignment():
    selftest.check_plane_assignment()


def test_inplane_transform_is_pure_reorder():
    selftest.check_inplane_transform()


def test_true_neighbour_triplets():
    selftest.check_true_neighbour_triplets()


def test_short_stack_padding():
    selftest.check_short_stack_padding()


def test_masked_pooling():
    selftest.check_masked_pooling()


def test_nan_target_rejected():
    selftest.check_nan_target_rejected()


def test_class_normalized_loss():
    selftest.check_class_normalization()


def test_epoch_loss_accounting():
    selftest.check_epoch_loss_accounting()


def test_single_class_auc_is_na():
    selftest.check_single_class_auc()


def test_label_join_by_key():
    selftest.check_label_join()


def test_empty_numeric_is_unknown():
    selftest.check_empty_numeric_is_unknown()


def test_unmentioned_weight_per_target():
    selftest.check_unmentioned_weight_per_target()


def test_frozen_reference_roundtrip(tmp_path):
    selftest.check_frozen_reference_roundtrip(tmp_path)


def test_soft_auc_matches_roc_auc():
    selftest.check_soft_auc_matches_roc_auc()


def test_soft_auc_bruteforce():
    selftest.check_soft_auc_bruteforce()


def test_soft_reference():
    selftest.check_soft_reference()


def test_bootstrap_keeps_soft():
    selftest.check_bootstrap_keeps_soft()


def test_crop_edge_fill():
    selftest.check_crop_edge_fill()


def test_foreground_extent_center():
    selftest.check_foreground_extent_center()


def test_patient_group_separation(tmp_path):
    selftest.check_group_separation(tmp_path)


def test_dataset_contract(cfg):
    selftest.check_dataset_contract(cfg)


def test_padding_invariance(cfg):
    selftest.check_padding_invariance(cfg)


def test_missing_slot_is_zero(cfg):
    selftest.check_missing_slot(cfg)


def test_masked_label_gradients(cfg):
    selftest.check_masked_label_gradients(cfg)


def test_gradient_accumulation(cfg):
    selftest.check_accumulation(cfg)


def test_sigmoid_not_softmax(cfg):
    selftest.check_sigmoid_outputs(cfg)


def test_window_loss_matches_full():
    selftest.check_window_loss_matches_full()


def test_window_weight_scaling():
    selftest.check_window_weight_scaling()


def test_window_empty_cases():
    selftest.check_window_empty_cases()


def test_global_loss_matches_formula():
    selftest.check_global_loss_matches_formula()


def test_global_no_dilution():
    selftest.check_global_no_dilution()


def test_attention_pool_starts_as_average(cfg):
    selftest.check_attention_pool_starts_as_average(cfg)


def test_spatial_pool_shapes(cfg):
    selftest.check_spatial_pool_shapes(cfg)


def test_focal_signal_survives_pooling(cfg):
    selftest.check_focal_signal_survives_pooling(cfg)


def test_spatial_pool_training_step(cfg):
    selftest.check_spatial_pool_training_step(cfg)


def test_float16_transport(cfg):
    selftest.check_float16_transport(cfg)


def test_explicit_run_selection():
    selftest.check_explicit_run_selection()


def test_eval_loader_budget(cfg):
    selftest.check_eval_loader_budget(cfg)


def test_global_training_step(cfg):
    selftest.check_global_training_step(cfg)


def test_window_training_step(cfg):
    selftest.check_window_training_step(cfg)


def test_epoch_reaches_workers(cfg):
    selftest.check_epoch_reaches_workers(cfg)


def test_augment_device_resolution(cfg):
    selftest.check_augment_device_resolution(cfg)


def test_augment_device_equivalence(cfg):
    selftest.check_augment_device_equivalence(cfg)


def test_checkpoint_roundtrip(cfg, tmp_path):
    selftest.check_checkpoint_roundtrip(cfg, tmp_path)


def test_train_step_reduces_loss(cfg):
    selftest.check_train_step_reduces_loss(cfg)


def test_pseudonymous_patient_id(tmp_path):
    selftest.check_pseudonymous_patient_id(tmp_path)


def test_ema_update_formula():
    selftest.check_ema_update_formula()


def test_ema_skipped_step():
    selftest.check_ema_skipped_step()


def test_ema_checkpoint_roundtrip(cfg):
    selftest.check_ema_checkpoint_roundtrip(cfg)


def test_ema_off_is_identity(cfg):
    selftest.check_ema_off_is_identity(cfg)


def test_exact_resume_with_ema(cfg):
    selftest.check_exact_resume_with_ema(cfg)


def test_laterality_rules():
    selftest.check_laterality_rules()


def test_laterality_derivation():
    selftest.check_laterality_derivation()


def test_laterality_dataset(cfg, tmp_path):
    selftest.check_laterality_dataset(cfg, tmp_path)


def test_depth_zones():
    selftest.check_depth_zones()


def test_target_attention_masking(cfg):
    selftest.check_target_attention_masking(cfg)


def test_target_attention_starts_as_head(cfg):
    selftest.check_target_attention_starts_as_head(cfg)


def test_target_attention_checkpoints(cfg, tmp_path):
    selftest.check_target_attention_checkpoints(cfg, tmp_path)


def test_target_attention_training_step(cfg):
    selftest.check_target_attention_training_step(cfg)
