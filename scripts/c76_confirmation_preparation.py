"""Shared gates for preparing reserved inputs; no numerical or label imports."""
from c75_port_utils import ROOT, read, sha
from c76_submission_contract import (BASE, BINDING, CONFIRM, CONSUMPTION, LEDGER,
    OPS_PLAN, POLICY, authorization, source_and_production, verify_files)

PREPARATION = CONFIRM / 'preparation'
CODE_PLAN = BASE / 'confirmation-inference-preparation-plan-r1.json'
DATASET = 'clarkkitchen/casmi26-c76-c13-inputs-r1-20261112'
PACKAGE = ROOT / 'outputs/publish/c76-c13-inputs-r1-20261112'
ORIGINAL = ROOT / 'outputs/publish/c13-mixed-casmi2022-20261110'
C73 = ROOT / 'outputs/local/c73-clean-target-validation-20261112'
ARMS = ('legacy', 'c76')
SLOTS = {arm: 'c76-c13-' + arm + '-r1' for arm in ARMS}
TEMPLATES = {'legacy': 'c73-full-masked-validation', 'c76': 'c76-full-masked-validation'}
CORE_PINS = {
    'legacy': '3f973abbb1593e356db4f6dec149c8ad99f5afa5ded8da8bab59152bd49e60e7',
    'c76': '182685722cb4bfb9c8bbdbabe1ec2a5ea46784d202d5692156a7faaf1171ffef'}


def prior():
    active = authorization()
    assert active['submissions_used'] == 0 and not LEDGER.exists()
    assert not CONSUMPTION.exists()
    # No query mapping, mask, dataset or inference preparation before admission.
    source_and_production()
    verify_files(read(OPS_PLAN)['files_sha256'])
    verify_files(read(BASE / 'confirmation-protocol-preparation-r1.json')['files_sha256'])
    plan = read(CODE_PLAN); verify_files(plan['files_sha256'])
    assert plan['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert plan['policy_sha256'] == sha(POLICY)
    assert plan['query_mapping_read'] is False and plan['frozen_before_C13_preparation']
    return plan


def read_queries():
    prior()
    import torch
    import pandas as pd
    path = ORIGINAL / 'validation-queries.parquet'; contract = read(ORIGINAL / 'validation-contract.json')
    assert sha(path) == contract['files'][path.name]
    q = pd.read_parquet(path)
    assert len(q) == q.molecule_id.nunique() == q.spectrum_id.nunique() == 120
    assert q.ionization_mode.value_counts().to_dict() == {'positive': 60, 'negative': 60}
    assert not {'key', 'smiles', 'normalized_smiles', 'inchikey14', 'formula'} & set(q.columns)
    support = read(ROOT / 'outputs/local/c59-c13-compatibility-20261112/channel-support-freeze.json')
    assert support['query_sha256'] == sha(path) and set(support['all_query_ids']) == set(q.molecule_id)
    assert len(support['pubchem_supported_ids']) == 119 and len(support['pubchem_unsupported_ids']) == 1
    assert q.set_index('molecule_id').loc[support['pubchem_unsupported_ids'], 'adduct'].tolist() == ['[M+FA-H]- ']
    return q
