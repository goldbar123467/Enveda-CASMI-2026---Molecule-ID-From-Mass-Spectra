"""Authenticated live rank evidence for the silver goal; never submits."""
import csv
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

COMP = 'enveda-CASMI26-molecule-id-mass-spectra'


def summarize(rows, username='clarkkitchen'):
    teams = len(rows); assert teams > 0
    cutoff = math.floor(teams * .05) if teams >= 1000 else (50 if teams >= 250 else math.floor(teams * .2))
    own = [row for row in rows if username in [s.strip().lower() for s in row['TeamMemberUserNames'].split(',')]]
    assert len(own) == 1, 'Own team must be unambiguous'
    boundary = next(row for row in rows if int(row['Rank']) == cutoff) if cutoff else None
    score = float(own[0]['Score'])
    tied = [int(row['Rank']) for row in rows if float(row['Score']) == score]
    return dict(teams=teams, team=own[0], nominal_silver_cutoff_rank=cutoff,
        cutoff_score=float(boundary['Score']) if boundary else None,
        own_score_rank_interval=[min(tied), max(tied)],
        provisional_silver_qualifying_position=int(own[0]['Rank']) <= cutoff,
        final_competition_medal=False,
        leaderboard_url=f'https://www.kaggle.com/competitions/{COMP}/leaderboard',
        medal_rule_url='https://www.kaggle.com/progression/competitions',
        limitation='Provisional live ranking; use actual rank, not a rounded score or a promised final medal')


def snapshot(api, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    api.competition_leaderboard_download(COMP, str(folder), quiet=True)
    archive = folder / (COMP + '.zip')
    with zipfile.ZipFile(archive) as contents:
        names = contents.namelist(); assert len(names) == 1 and names[0].endswith('.csv')
        rows = list(csv.DictReader(io.StringIO(contents.read(names[0]).decode('utf-8-sig'))))
    report = summarize(rows)
    report.update(checked_utc=datetime.now(timezone.utc).isoformat(), source_csv=names[0],
                  source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (folder / 'leaderboard.json').write_text(json.dumps(report, indent=2, ensure_ascii=True))
    return report
