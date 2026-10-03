"""Explicit C.Origami cache and CTCF-input identity; no inference."""
import json
import os
from pathlib import Path

CELLS = ('GM12878', 'K562', 'HepG2', 'HCT116')
KNOWN_MODELS = {
    'corigami_bench_v2_offctcf': 'released',
    'corigami_bench_v2_ctcfSaitou': 'retrained_53p5b',
    'corigami_bench_v2_humanctcf30M': 'released',
}
KNOWN_PROTOCOLS = {
    'corigami_bench_v2_offctcf': 'unsampled_bam',
    'corigami_bench_v2_ctcfSaitou': 'unsampled_bam',
    'corigami_bench_v2_humanctcf30M': 'humanctcf30M',
}
PROTOCOLS = {
    'humanctcf30M': 'Aligned IP/control BAM subsampling to approximately 30 million reads (samtools seed 2021), retaining lower-depth inputs; RPKM 1-bp coverage and bigwigCompare',
    'unsampled_bam': 'Historical IP/control BAM RPKM 1-bp coverage and bigwigCompare before 30M subsampling',
    'legacy_original': 'Historical original C.Origami input preparation',
}


def resolve_stores(env=None, pool_root=None):
    """Validate explicit weight/input declarations and selected cache availability.

    GM12878 retains the original official CTCF input in both weight arms. Any
    CO_GM12878_STORE override must contain predictions from the selected weights.
    Paths/declarations do not independently verify cache-generating execution.
    """
    env = os.environ if env is None else env
    legacy = env.get('XCHROM_BENCH_LEGACY') == '1'
    name = env.get('CO_STORE')
    if not name:
        if not legacy:
            raise ValueError('Set CO_STORE explicitly; no historical cache is selected.')
        if pool_root is None:
            raise ValueError('Legacy cache fallback requires an explicit pool root.')
        name = str(Path(pool_root) / 'corigami_bench_v2')
    root = Path(name).expanduser().absolute()
    historical_original = root.name == 'corigami_bench_v2'
    if historical_original and not legacy:
        raise ValueError('corigami_bench_v2 is historical; use the intended CO_STORE explicitly.')
    inferred = KNOWN_MODELS.get(root.name, 'legacy_original' if historical_original else None)
    inferred_protocol = KNOWN_PROTOCOLS.get(root.name, 'legacy_original' if historical_original else None)
    if not inferred and not env.get('CO_MODEL'):
        raise ValueError('Unknown CO_STORE name: set CO_MODEL explicitly to released or retrained_53p5b.')
    if not inferred_protocol and not env.get('CO_CTCF_PROTOCOL'):
        raise ValueError('Unknown CO_STORE name: set CO_CTCF_PROTOCOL explicitly to the producer-declared input protocol.')
    model = env.get('CO_MODEL', inferred)
    protocol = env.get('CO_CTCF_PROTOCOL', inferred_protocol)
    if model not in ('released', 'retrained_53p5b', 'legacy_original'):
        raise ValueError('CO_MODEL must be released or retrained_53p5b (legacy_original requires legacy mode).')
    if (model == 'legacy_original' or protocol == 'legacy_original') and not legacy:
        raise ValueError('legacy_original requires XCHROM_BENCH_LEGACY=1.')
    if protocol not in PROTOCOLS:
        raise ValueError(f'CO_CTCF_PROTOCOL must be one of {tuple(PROTOCOLS)}.')
    if inferred and model != inferred:
        raise ValueError(f'CO_MODEL={model} conflicts with the documented {root.name} cache ({inferred}).')
    if inferred_protocol and protocol != inferred_protocol:
        raise ValueError(f'CO_CTCF_PROTOCOL={protocol} conflicts with the documented {root.name} cache ({inferred_protocol}).')
    selected = tuple(c.strip() for c in env.get('CELLS', ','.join(CELLS)).split(',') if c.strip())
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(CELLS):
        raise ValueError(f'CELLS must be a nonempty unique subset of {CELLS}.')
    stores = {cell: root for cell in CELLS}
    if env.get('CO_GM12878_STORE'):
        gm = Path(env['CO_GM12878_STORE']).expanduser().absolute()
        gm_model = KNOWN_MODELS.get(gm.name, 'released' if gm.name == 'corigami_bench_v2' else None)
        weight_arm = 'released' if model == 'legacy_original' else model
        if gm_model and gm_model != weight_arm:
            if model == 'retrained_53p5b':
                raise ValueError('Retrained GM12878 requires retrained predictions, not the released-model cache.')
            raise ValueError('Released GM12878 requires released predictions, not the retrained-model cache.')
        stores['GM12878'] = gm
    for cell in selected:
        source = stores[cell]
        if not source.is_dir() or not any(source.glob(f'{cell}_chr*.maps.npz')):
            extra = ' Set CO_GM12878_STORE to its matching-weight official-input cache if stored separately.' if cell == 'GM12878' else ''
            raise ValueError(f'No C.Origami cache files for {cell} in {source}.{extra}')
    return {
        'model': model,
        'key': 'corigami_retrained_53p5b' if model == 'retrained_53p5b' else 'corigami',
        'label': 'C.Origami retrained (53.5 B)' if model == 'retrained_53p5b' else 'C.Origami 1m',
        'stores': {cell: str(path) for cell, path in stores.items()},
        'selected_cells': list(selected),
        'ctcf_protocol': protocol,
        'ctcf_sources': {cell: ('original official GM12878 input' if cell == 'GM12878' else PROTOCOLS[protocol]) for cell in selected},
        'scope': 'Explicit cache selection and file availability; generating model/input identity remains producer provenance.',
    }


def main():
    print(json.dumps(resolve_stores(), indent=2))


if __name__ == '__main__': main()
