"""--loss and --label-smoothing must reach every grad_cache_loss call in training.

grad_cache_loss falls back to plain clip_loss when no loss_fn is passed, so a call that
leaves it out silently trains with InfoNCE whatever the flags say. Checks the source,
so it needs no GPU, model or data:

    pytest train/tests/test_loss_fn_wiring.py -v
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "train_contrastors.py"


def grad_cache_calls_in_main():
    tree = ast.parse(SOURCE.read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    return [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "grad_cache_loss"
    ]


def test_training_loops_call_grad_cache_loss():
    # Text warmup and the main loop both train through grad_cache_loss.
    assert len(grad_cache_calls_in_main()) >= 2


def test_every_grad_cache_call_passes_the_selected_loss():
    missing = [
        call.lineno
        for call in grad_cache_calls_in_main()
        if not any(
            kw.arg == "loss_fn" and isinstance(kw.value, ast.Name) and kw.value.id == "active_loss_fn"
            for kw in call.keywords
        )
    ]
    assert not missing, (
        f"grad_cache_loss called without loss_fn=active_loss_fn at "
        f"train_contrastors.py line(s) {missing}; --loss and --label-smoothing are ignored there"
    )
