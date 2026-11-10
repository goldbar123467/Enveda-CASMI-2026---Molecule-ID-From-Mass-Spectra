"""Plain-text CASMI2022 MSP parsing and label-blind scan assignment.

Window/max-intensity method follows the MassFormer authors' preparation source.
Unlike its empty-mask fallback, an unmatched target here has no assigned scan.
No downloaded code or pickle is executed.
"""
import math


def parse_msp(stream):
    headers, peaks, expected = {}, [], None

    def finish():
        if expected is None or len(peaks) != expected:
            raise ValueError('MSP peak count does not match Num peaks')
        return dict(name=headers.get('name', ''),
                    precursor_mz=float(headers.get('precursormz', 'nan')),
                    rt=float(headers.get('retentiontime', 'nan')),
                    spectrum_type=headers.get('spectrum_type', ''), peaks=peaks)

    for raw in stream:
        line = raw.strip()
        if not line:
            if headers and expected is not None and len(peaks) == expected:
                yield finish()
                headers, peaks, expected = {}, [], None
            continue
        if line.lower().startswith('name:') and headers:
            yield finish()
            headers, peaks, expected = {}, [], None
        if expected is not None and len(peaks) < expected:
            values = line.split()
            if len(values) < 2:
                raise ValueError('Malformed MSP peak')
            peaks.append((float(values[0]), float(values[1])))
        else:
            key, separator, value = line.partition(':')
            if not separator:
                raise ValueError('Malformed MSP header or excess peak')
            key = key.strip().lower().replace(' ', '_')
            headers[key] = value.strip()
            if key == 'num_peaks':
                expected = int(value)
                if not 0 <= expected <= 100000:
                    raise ValueError('Unexpected MSP peak count')
    if headers:
        yield finish()


def scan_rank(scan, precursor_mz, rt):
    """Return a deterministic best-first key, or None outside both windows."""
    if scan['spectrum_type'] != 'MS2':
        return None
    error_mz, error_rt = abs(scan['precursor_mz'] - precursor_mz), abs(scan['rt'] - rt)
    if not math.isfinite(error_mz + error_rt) or error_mz > .005 or error_rt > .1:
        return None
    total = sum(intensity for mz, intensity in scan['peaks']
                if math.isfinite(mz) and math.isfinite(intensity) and mz > 0 and intensity > 0)
    return -total, error_rt, error_mz, scan['name']
