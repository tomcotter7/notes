# Transformers (The Package)

## Hooks

You can register hooks to the fwd & bwd passes of the model as follows:

```python
# NaN diagnostic: log the first layer that produces NaN/Inf in forward or backward
    def _make_fwd_hook(name):
        def hook(module, input, output):
            tensors = [output] if isinstance(output, torch.Tensor) else []
            if isinstance(output, tuple):
                tensors = [t for t in output if isinstance(t, torch.Tensor)]
            for t in tensors:
                if t.isnan().any() or t.isinf().any():
                    nan_count = t.isnan().sum().item()
                    inf_count = t.isinf().sum().item()
                    logger.warning(
                        f"FWD NaN/Inf in: {name} | shape={t.shape} | "
                        f"nan={nan_count} inf={inf_count} min={t[t.isfinite()].min().item() if t.isfinite().any() else 'N/A'} "
                        f"max={t[t.isfinite()].max().item() if t.isfinite().any() else 'N/A'}"
                    )
                    break

        return hook

    def _make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_output):
                if g is not None and (g.isnan().any() or g.isinf().any()):
                    nan_count = g.isnan().sum().item()
                    inf_count = g.isinf().sum().item()
                    logger.warning(
                        f"BWD NaN/Inf in grad_output[{i}] of: {name} | shape={g.shape} | "
                        f"nan={nan_count} inf={inf_count}"
                    )
                    break

        return hook

    for name, mod in model.named_modules():
        mod.register_forward_hook(_make_fwd_hook(name))
        mod.register_full_backward_hook(_make_bwd_hook(name))
    logger.info("Registered NaN detection hooks (fwd + bwd) on all modules")
```
