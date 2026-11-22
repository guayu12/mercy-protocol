# avatar.vrm

`AvatarSample_B` by VRoid Project, (c) pixiv Inc. — https://vroid.pixiv.help/hc/ja/articles/4402394424089-AvatarSample-A-B-C

License: [VRM Public License 1.0](https://vrm.dev/licenses/1.0/) — per the model's embedded metadata:
redistribution allowed, modification and redistribution allowed, commercial use by corporations allowed, credit not required.

Modification: embedded textures re-encoded to cut file size (20.5 MB -> 13.6 MB): the 2048px thumbnail is a 512px JPEG,
opaque RGB textures are JPEG q92, the rest are re-optimized PNG. Geometry, materials, expressions and spring bones are unchanged.

To use your own character, export a VRM (e.g. from [VRoid Studio](https://vroid.com/en/studio)) and replace `avatar.vrm`.
VRM 0.x and 1.0 files both work.

# motions/

| File | Source | License |
|---|---|---|
| `motions/idle_loop.vrma` | [pixiv/ChatVRM](https://github.com/pixiv/ChatVRM) `public/idle_loop.vrma` (archived repo) | MIT, Copyright (c) 2023 pixiv Inc. — full text in [`motions/LICENSE-ChatVRM.txt`](motions/LICENSE-ChatVRM.txt) |
| `motions/ual_idle.vrma` | Quaternius [Universal Animation Library](https://quaternius.com/packs/universalanimationlibrary.html) (Standard, [itch.io](https://quaternius.itch.io/universal-animation-library)), clip `Idle_Loop` from `Unreal-Godot/UAL1_Standard.glb` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `motions/ual_talking.vrma` | Same pack, clip `Idle_Talking_Loop` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

Gestures `greet` / `think` are keyframed in code (`src/avatar/motion.ts`); `nod` / `shake` are procedural. The two `ual_*.vrma` files were converted offline from the GLB (T-pose rest, 30 fps resample) with `frontend/scripts/ual2vrma.mjs`; there is no runtime retargeting. Source ZIP SHA-256: `cc73fc4e495b82958207316596317a3f40b9fa38065bde1027937452da537724`. No other motion assets are shipped.
