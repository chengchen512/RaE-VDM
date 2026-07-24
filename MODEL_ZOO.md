# Model Zoo

Pretrained weights are intentionally excluded from the source archive and Git
history. Publish them through a release asset, institutional repository, or
model hosting service, then add the download URLs and SHA-256 hashes below.

| File | Purpose | Included |
|---|---|---|
| `FINAL_BEST_INFERENCE.pt` | Final EMA inference checkpoint | No |
| `raft_small_C_T_V2-01064c6d.pth` | Official torchvision RAFT-Small weights | No |

Recommended placement for local examples:

```text
RaE-VDM/
  FINAL_BEST_INFERENCE.pt
  raft_small_C_T_V2-01064c6d.pth
```

The final RaE-VDM checkpoint should be loaded with
`configs/inference.yaml`. The inference code prefers `model_ema`
when that key is present.
