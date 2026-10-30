# Third-party code and data

| Component | Location | Source | Licence |
|---|---|---|---|
| arim (Bristol NDT group) | `10_ActivePAUT代码/arim_api/` | https://github.com/ndtatbristol/arim, commit 0cee9de26ae995ca80e00bc67801ffb3b1188e44 (see `COMMIT.txt`) | MIT (`LICENSE.txt` kept) |
| SimNDT-modern | `02_SimNDT弹性波仿真/SimNDT-modern/` | Own P-SV phased-array rewrite built on the EFIT approach of SimNDT (https://github.com/mmolero/SimNDT) | See note |

Note on SimNDT-modern: the upstream SimNDT repository does not state a licence on its README. Before making this
repository public, confirm the upstream terms or replace this directory with a link and install instructions.

External datasets are NOT included (download them yourself, check each licence):

- Strathclyde FMC dataset, Lardner 2015 (CC BY 4.0): used by `apsim/contact.py`, `run_real_lardner*.py`
- Bristol multi-view data set (data.bris, `2plergoban2sh2dhqy1q122e1d`) and MCMC notch data: used by `apsim/immersion.py`, `run_bristol_sdh.py`, `run_real_notch.py`

Expected location: `05_实测数据分析/<dataset folder>/` next to `10_ActivePAUT代码/` (see the `DATA` constants in `apsim/contact.py` and `apsim/immersion.py`).
