"""Project SD fields onto native physical G, then sample or lift that projection.

This is NOT lossless interpolation of raw nonlinear HXC. For wavefunctions cut
at Ecut, |(k+G)-(k+G')| <= 2 sqrt(Ecut), so only potential Fourier coefficients
with |G-G'|^2 <= 4 Ecut enter the finite-PW local matrix. Retaining those
coefficients preserves that matrix, while changing the pointwise potential.
The native SEED_CHARGE.dat must contain that complete physical reciprocal set.
"""
from pathlib import Path
import argparse
import json
import math
import struct
import tempfile

import numpy as np

COARSE = (81, 243, 225)
FINE = (243, 243, 225)


def read_field(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with path.open('rb') as f:
        header = struct.unpack('<7i', f.read(28))
        shape = header[1:5]
        size = math.prod(shape) * 8
        assert header[0] == header[5] == 16 and header[6] == size
        assert shape[3] == 4 and all(n > 0 for n in shape)
        assert path.stat().st_size == size + 32
        f.seek(-4, 2)
        assert struct.unpack('<i', f.read(4))[0] == size
    return np.memmap(path, dtype='<f8', mode='r', offset=28, shape=shape, order='F'), shape[:3]


def read_millers(path: Path) -> tuple[np.ndarray, dict]:
    with path.open('rb') as f:
        def record() -> bytes:
            size = struct.unpack('<i', f.read(4))[0]
            assert 0 <= size <= path.stat().st_size
            data = f.read(size)
            assert len(data) == size and struct.unpack('<i', f.read(4))[0] == size
            return data

        gamma, count, nspin = struct.unpack('<3i', record())
        assert gamma == 0 and nspin == 4 and 0 < count <= math.prod(COARSE)
        bg = np.frombuffer(record(), '<f8').reshape(3, 3, order='F')
        mill = np.frombuffer(record(), '<i4').reshape(3, count, order='F').T.astype(np.int64)
    assert np.isfinite(bg).all()
    return mill, {'native_G_count': count, 'reciprocal_bg': bg.tolist(),
                  'Miller_min': mill.min(axis=0).tolist(), 'Miller_max': mill.max(axis=0).tolist()}


def indices(mill: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    return tuple((mill % np.array(shape)).T)


def validate_millers(mill: np.ndarray, coarse: tuple[int, ...], fine: tuple[int, ...]) -> None:
    assert np.all(np.abs(mill) <= (np.array(fine) - 1) // 2)
    for shape in (coarse, fine):
        keys = np.ravel_multi_index(indices(mill, shape), shape)
        assert len(np.unique(keys)) == len(mill), ('Aliased physical G vectors', shape)
        negative = np.ravel_multi_index(indices(-mill, shape), shape)
        assert np.array_equal(np.sort(keys), np.sort(negative)), 'G set is not conjugate closed'
    assert np.count_nonzero(np.all(mill == 0, axis=1)) == 1


def spectrum(values: np.ndarray) -> np.ndarray:
    assert np.isfinite(values).all()
    return np.fft.fftn(values) / values.size


def from_coefficients(coeff: np.ndarray, mill: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    full = np.zeros(shape, dtype=complex)
    full[indices(mill, shape)] = coeff
    values = np.fft.ifftn(full) * math.prod(shape)
    error = float(np.max(np.abs(values.imag)))
    assert error <= 1e-11 * max(1., float(np.max(np.abs(values.real)))), ('Complex field', error)
    return values.real


def difference(a: np.ndarray, b: np.ndarray) -> dict:
    delta = a - b
    return {'max_abs': float(np.max(np.abs(delta))), 'rms': float(np.sqrt(np.mean(np.abs(delta) ** 2)))}


def convert(values: np.ndarray, mill: np.ndarray, mode: str,
            coarse: tuple[int, ...] = COARSE, fine: tuple[int, ...] = FINE) -> tuple[np.ndarray, dict]:
    expected = fine if mode == 'restrict' else coarse
    assert mode in ('restrict', 'lift') and values.shape == expected
    spec = spectrum(values)
    coeff = spec[indices(mill, expected)].copy()
    spec[indices(mill, expected)] = 0.
    discarded = float(np.linalg.norm(spec.ravel()))
    retained = float(np.linalg.norm(coeff))
    del spec
    projected_fine = from_coefficients(coeff, mill, fine)
    stride = np.array(fine) // np.array(coarse)
    assert np.array_equal(stride * coarse, fine)
    sample = projected_fine[tuple(slice(None, None, int(n)) for n in stride)]
    # Re-extract from actual coarse samples, not the original coefficients.
    sample_spec = spectrum(sample)
    back_coeff = sample_spec[indices(mill, coarse)].copy()
    del sample_spec
    error = difference(back_coeff, coeff)
    metrics = {'discarded_spectral_l2_RMS_units': discarded,
               'discarded_spectral_relative_l2': discarded / max(math.hypot(retained, discarded), 1e-300),
               'in_sphere_coefficient_error': error,
               'in_sphere_coefficient_relative_l2': float(np.linalg.norm(back_coeff - coeff)) / max(retained, 1e-300)}
    if mode == 'restrict':
        lifted = from_coefficients(back_coeff, mill, fine)
        metrics['raw_fine_vs_lifted'] = difference(values, lifted)
        metrics['projected_fine_vs_lifted'] = difference(projected_fine, lifted)
        return sample, metrics
    metrics['coarse_input_vs_restricted_output'] = difference(values, sample)
    return projected_fine, metrics


def process(mode: str, source: Path, seed: Path, output: Path, metrics_path: Path) -> dict:
    assert not output.exists() and not output.is_symlink(), output
    assert not metrics_path.exists() and not metrics_path.is_symlink(), metrics_path
    assert output.resolve() != metrics_path.resolve()
    mill, details = read_millers(seed)
    validate_millers(mill, COARSE, FINE)
    raw, shape = read_field(source)
    assert shape == (FINE if mode == 'restrict' else COARSE)
    target = COARSE if mode == 'restrict' else FINE
    size = math.prod(target) * 4 * 8
    result = {'physical_release': False, 'operation': mode, 'source': str(source.resolve()),
              'physical_G_source': str(seed.resolve()), 'output': str(output.resolve()),
              'input_grid': list(shape), 'output_grid': list(target), **details, 'components': [],
              'meaning': 'Explicit physical-G spectral projection; raw nonlinear HXC is not assumed bandlimited.',
              'metric_normalization': 'FFTN/N; coefficient L2 equals real-space RMS by Parseval.',
              'limits': ['G set completeness and 4*ecutwfc provenance require the matching QE reference.',
                         'Numerical reconstruction does not validate density fitting or magnetic response.']}
    with output.open('xb') as f:
        f.write(struct.pack('<7i', 16, *target, 4, 16, size))
        for component in range(4):
            values, metrics = convert(raw[..., component], mill, mode)
            f.write(np.asarray(values, dtype='<f8').tobytes(order='F'))
            result['components'].append({'component': component, **metrics})
        f.write(struct.pack('<i', size))
    actual, actual_shape = read_field(output)
    assert actual_shape == target and np.isfinite(actual).all()
    with metrics_path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    return result


def self_check() -> None:
    coarse, fine = (81, 243, 1), (243, 243, 1)
    m = np.array([50, -60, 0])
    mill = np.array([[0, 0, 0], m, -m])
    validate_millers(mill, coarse, fine)
    x = np.arange(243)[:, None, None] / 243
    y = np.arange(243)[None, :, None] / 243
    oracle = np.cos(2 * np.pi * (m[0] * x + m[1] * y))
    # An out-of-sphere mode shares coarse residues with m, so restricting AFTER
    # downsampling would corrupt the very in-sphere coefficient being kept.
    raw = oracle + .2 * np.cos(2 * np.pi * (-31 * x - 60 * y))
    samples, metrics = convert(raw, mill, 'restrict', coarse, fine)
    exact, lift_metrics = convert(samples, mill, 'lift', coarse, fine)
    assert np.max(np.abs(exact - oracle)) < 2e-12
    assert abs(metrics['discarded_spectral_l2_RMS_units'] - .2 / math.sqrt(2)) < 1e-13
    bad, _ = convert(raw[::3], mill, 'lift', coarse, fine)
    assert np.max(np.abs(bad - oracle)) > .19
    wrong_mill = (mill + np.array(coarse) // 2) % coarse - np.array(coarse) // 2
    naive = from_coefficients(spectrum(samples)[indices(mill, coarse)], wrong_mill, fine)
    assert np.max(np.abs(naive[::3] - samples)) < 2e-12
    assert np.max(np.abs(naive - oracle)) > 1.
    # Exercise the exact binary record contract with real four-component data.
    with tempfile.TemporaryDirectory() as tmp:
        seed = Path(tmp) / 'SEED_CHARGE.dat'
        with seed.open('xb') as f:
            for data in (struct.pack('<3i', 0, len(mill), 4),
                         np.eye(3, dtype='<f8').tobytes(order='F'),
                         np.asarray(mill.T, dtype='<i4').tobytes(order='F')):
                f.write(struct.pack('<i', len(data)) + data + struct.pack('<i', len(data)))
        parsed, _ = read_millers(seed)
        assert np.array_equal(parsed, mill)
        path = Path(tmp) / 'field.bin'
        values = np.stack([oracle, 2 * oracle, -oracle, np.zeros_like(oracle)], axis=-1)
        with path.open('xb') as f:
            f.write(struct.pack('<7i', 16, *fine, 4, 16, values.nbytes))
            f.write(values.tobytes(order='F'))
            f.write(struct.pack('<i', values.nbytes))
        readback, shape = read_field(path)
        assert shape == fine and np.array_equal(readback, values)
    print(json.dumps({'self_check': 'PASS', 'physical_release': False,
                      'physical_lift_error': difference(exact, oracle),
                      'naive_fine_error': difference(naive, oracle),
                      'naive_coarse_roundtrip_error': difference(naive[::3], samples),
                      'projection_metrics': metrics, 'lift_metrics': lift_metrics}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('restrict', 'lift', 'self-check'))
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('seed_charge', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--metrics', type=Path)
    args = parser.parse_args()
    if args.mode == 'self-check':
        assert args.input is args.seed_charge is args.output is args.metrics is None
        self_check()
    else:
        if None in (args.input, args.seed_charge, args.output):
            parser.error('restrict/lift requires INPUT SEED_CHARGE.dat OUTPUT')
        report = args.metrics or args.output.with_name(args.output.name + '.metrics.json')
        print(json.dumps(process(args.mode, args.input, args.seed_charge, args.output, report), indent=2))
