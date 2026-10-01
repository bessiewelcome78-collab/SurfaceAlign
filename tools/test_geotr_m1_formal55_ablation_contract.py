#!/usr/bin/env python3
"""Fast CPU contract for FORMAL55 opt-in M1 ablations."""

from types import SimpleNamespace as NS

import torch

from utils.geotr_m1_loss import compute_geotr_m1_loss
from utils.geotr_m1_transport import ExactGeometryTransportSegmenter


def cfg(**overrides):
    values = dict(
        MHCS_HIDDEN_DIM=32,
        SEMANTIC_CHANNELS=8,
        MHCS_TEXT_DIM=8,
        GEOTOPO_FLOW_INIT_SCALE_PX=1.0,
    )
    values.update(overrides)
    return NS(M1=NS(**values))


def inputs():
    return dict(
        base_logits=torch.randn(2, 1, 24, 24),
        image=torch.randn(2, 3, 24, 24),
        semantic_map=torch.randn(2, 8, 12, 12),
        text_features=torch.randn(2, 8),
    )


def make_nonidentity(model):
    with torch.no_grad():
        model.mean_head.flow_out.weight.normal_(0.0, 0.03)
        model.context_gate.fill_(0.7)


def main():
    torch.manual_seed(7)
    legacy = ExactGeometryTransportSegmenter(cfg()).eval()
    explicit = ExactGeometryTransportSegmenter(
        cfg(
            GEOTR_M1_USE_SEMANTIC_CONDITIONING=True,
            GEOTR_M1_USE_TEXT_CONDITIONING=True,
            GEOTR_M1_USE_ANCHOR_CUES=True,
            GEOTR_M1_TRANSPORT_SPACE="logit",
        )
    ).eval()
    make_nonidentity(legacy)
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    batch = inputs()
    with torch.no_grad():
        _, old_aux = legacy.generate(**batch)
        _, new_aux = explicit.generate(**batch)
    assert torch.equal(old_aux["geotopo_final_probs"], new_aux["geotopo_final_probs"])
    assert torch.equal(old_aux["geotopo_geometry_flow_px"], new_aux["geotopo_geometry_flow_px"])

    no_text = ExactGeometryTransportSegmenter(
        cfg(GEOTR_M1_USE_TEXT_CONDITIONING=False)
    ).eval()
    no_text.load_state_dict(legacy.state_dict(), strict=True)
    changed_text = dict(batch)
    changed_text["text_features"] = batch["text_features"] + 100.0
    with torch.no_grad():
        _, first = no_text.generate(**batch)
        _, second = no_text.generate(**changed_text)
    assert torch.equal(first["geotopo_final_probs"], second["geotopo_final_probs"])

    no_semantic = ExactGeometryTransportSegmenter(
        cfg(GEOTR_M1_USE_SEMANTIC_CONDITIONING=False)
    ).eval()
    no_semantic.load_state_dict(legacy.state_dict(), strict=True)
    changed_semantic = dict(batch)
    changed_semantic["semantic_map"] = batch["semantic_map"] + 100.0
    with torch.no_grad():
        _, first = no_semantic.generate(**batch)
        _, second = no_semantic.generate(**changed_semantic)
    assert torch.equal(first["geotopo_final_probs"], second["geotopo_final_probs"])

    # The transferable image+anchor core must not require UniMedCLIP spatial
    # features or a text vector once both optional conditioners are disabled.
    core = ExactGeometryTransportSegmenter(
        cfg(
            GEOTR_M1_USE_SEMANTIC_CONDITIONING=False,
            GEOTR_M1_USE_TEXT_CONDITIONING=False,
        )
    ).eval()
    core.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        _, core_aux = core.generate(
            base_logits=batch["base_logits"],
            image=batch["image"],
            semantic_map=None,
            text_features=None,
        )
    assert torch.isfinite(core_aux["geotopo_final_probs"]).all()

    # Legacy direct-use behavior remains available only when explicitly
    # requested.  CAUSAL56 and all new formal configs use the safer default.
    isolated = inputs()
    isolated["base_logits"].requires_grad_(True)
    isolated["semantic_map"].requires_grad_(True)
    isolated["text_features"].requires_grad_(True)
    train_model = ExactGeometryTransportSegmenter(
        cfg(GEOTR_M1_DETACH_CONDITIONERS=False)
    )
    make_nonidentity(train_model)
    _, isolated_aux = train_model.generate(**isolated)
    isolated_aux["geotopo_final_probs"].mean().backward()
    assert isolated["base_logits"].grad is None
    for key in ("semantic_map", "text_features"):
        assert isolated[key].grad is not None, (key, isolated[key].grad)
    assert train_model.mean_head.flow_out.weight.grad is not None
    assert float(train_model.mean_head.flow_out.weight.grad.abs().sum()) > 0.0

    probability = ExactGeometryTransportSegmenter(
        cfg(GEOTR_M1_TRANSPORT_SPACE="probability")
    ).eval()
    probability.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        _, aux = probability.generate(**batch)
    assert torch.isfinite(aux["geotopo_final_logits"]).all()
    assert (aux["geotopo_final_probs"] > 0).all() and (aux["geotopo_final_probs"] < 1).all()

    masks = torch.randint(0, 2, (2, 24, 24)).float()
    edge_cfg = cfg(
        GEOTR_M1_DEFORM_MODE="edge_aware",
        GEOTOPO_SMOOTHNESS_WEIGHT=0.001,
        GEOTR_M1_FOLDING_WEIGHT=0.001,
        GEOTR_M1_MIN_JACOBIAN=0.05,
    )
    edge_model = ExactGeometryTransportSegmenter(edge_cfg)
    make_nonidentity(edge_model)
    edge_logits, edge_aux = edge_model.generate(**batch)
    edge_loss, edge_diag = compute_geotr_m1_loss(
        edge_cfg, edge_logits, masks, edge_aux
    )
    edge_loss.backward()
    assert torch.isfinite(edge_loss)
    assert float(edge_diag["geotr_m1_deform_mode_id"]) == 2.0
    assert 0.0 <= float(edge_diag["geotr_m1_edge_band_fraction"]) <= 1.0
    print(
        "[PASS] FORMAL55.1 identity, transferable core, factual Base detach, "
        "edge-aware deformation and opt-in ablation contracts"
    )


if __name__ == "__main__":
    main()
