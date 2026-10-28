"""Freeze the existing SP+ implementation and preregister the default settings.

Only used once during setup. Reproduction uses the delivered frozen files.
"""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = root.parent/'broadcast_reconstruction'/'plot_figure.py'
s = source.read_text(encoding='utf-8')
f = next(x for x in ast.parse(s).body if isinstance(x, ast.FunctionDef) and x.name == 'compact_sp_plus')
(root/'reference_sp_plus.py').write_text('"""Frozen verbatim function from ../broadcast_reconstruction/plot_figure.py.\nSee source_audit.md and provenance.json. No data-generation code imported.\n"""\nimport numpy as np\n\n'+ast.get_source_segment(s,f)+'\n', encoding='utf-8')
settings = []
ranks = [8,8,12,12,16,16,20,24,28,32]
for eta in [0, .1, .3, 1.]:
    settings.append(dict(id=f'shared_R32_eta{eta:g}', generator='shared_reference', eta=eta, R=32, role='main' if eta==.3 else 'supplement'))
settings.append(dict(id='independent_R32',generator='independent',eta=None,R=32,role='supplement'))
for R in [16,64]:
    settings.append(dict(id=f'shared_R{R}_eta0.3',generator='shared_reference',eta=.3,R=R,role='sensitivity'))
for item in settings:
    item.update(m=512,n=512,ranks=[max(1, int(r*item['R']/32+.5)) for r in ranks],weights=[.1]*10)
config=dict(schema_version=1, experiment='synthetic RBLA aggregation experiment', dtype='float64', effective_lora_scaling=1., seeds=list(range(2026091100,2026091130)), main_setting='shared_R32_eta0.3', settings=settings, blas_threads=1, save_global_factors=True, post_aggregation_gauge=False, tolerance=dict(normalization_floor='64*eps*||B||F*||A||F', absolute_error='128*eps*||B||F*||A||F + 1e-12*||X||F', presentation='Only negative excess within the absolute tolerance may be shown as zero'), generator=dict(reference='iid Gaussian: B0 std=1/sqrt(m); A0 std=1/sqrt(n)', noise='iid Gaussian scaled by corresponding reference block RMS; expected noise Frobenius norm squared equals reference norm squared', rng='NumPy PCG64; SeedSequence([seed,R,m,n]).spawn(1+2*N); shared random streams across eta settings', rank_mapping='nearest integer, half-up: floor(base_rank*R/32+0.5), minimum 1'))
(root/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
(root/'provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),function='compact_sp_plus',function_sha256=hashlib.sha256(ast.get_source_segment(s,f).encode()).hexdigest(),git_head='8806e1f48d21053fcc3aed661a65edbd5bfd2aea',source_was_locally_modified=True,agents_md_found=False),indent=2),encoding='utf-8')
print('Frozen existing SP+ function and fixed 7 settings x 30 seeds before running.')
