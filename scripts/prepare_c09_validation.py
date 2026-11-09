"""Derive an acquisition-disjoint C09 confirmation panel from audited sources.

Reuse the C08 export implementation without changing that historical source.
Exclude all previously used C08 identities AND all cached input groups connected
to competition training. Use every remaining eligible identity, no score filter.
"""
import ast,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def source():
    original=(ROOT/'scripts/prepare_compatible_validation.py').read_text(encoding='utf-8')
    replacements={
        "out=ROOT/'outputs/publish/c08-validation-20261108'":"out=ROOT/'outputs/publish/c09-validation-20261109'",
        "eligible=mol.loc[~mol.key.isin(forbidden)].copy();assert len(eligible)>=128":"""# Reconstruct the exact combined catalogue order used by the saved
    # cross-label acquisition groups; exclude whole groups, not only keys.
    combined=pd.concat([base[['key']],mol.loc[~mol.key.isin(set(base.key)),['key']]],ignore_index=True)
    saved=np.load(ROOT/'outputs/local/long-20261106/spectrum_groups.npz',allow_pickle=False)
    assert len(combined)==len(saved['groups']) and combined.key.is_unique
    group_lookup=dict(zip(combined.key,saved['groups']))
    base_groups=set(saved['groups'][:len(base)])
    previous=pd.read_csv(ROOT/'outputs/publish/c08-validation-20261108/labels.csv',dtype={'key':str})
    forbidden.update(previous.key)
    eligible=mol.loc[~mol.key.isin(forbidden)].copy()
    eligible['input_group']=eligible.key.map(group_lookup)
    assert eligible.input_group.notna().all()
    eligible=eligible[~eligible.input_group.isin(base_groups)]
    # Remove cross-label aliases inside the external panel as well.
    eligible=eligible[~eligible.input_group.duplicated(keep=False)].copy()
    assert len(eligible)>=50, 'Too few acquisition-disjoint identities'
    group_pin=sha(ROOT/'outputs/local/long-20261106/spectrum_groups.npz')""",
        "'C08-external-panel-v1:'":"'C09-acquisition-disjoint-confirmation-v1:'",
        "selected=pd.concat([g.sort_values('draw').head(64) for _,g in eligible.groupby('source')]).sort_values('draw').head(128)":"selected=eligible.sort_values('draw',kind='stable').copy()",
        "assert len(selected)==128 and selected.key.is_unique":"assert selected.key.is_unique and selected.input_group.is_unique",
        "'C08V_'":"'C09V_'",
        "selected['partition']=['development' if i%2==0 else 'confirmation' for i in range(len(selected))]":"selected['partition']='confirmation'",
        "'C08-view:'":"'C09-view:'",
        "'C08S_'":"'C09S_'",
        "contract=dict(version=1,queries=128,spectra=len(queries),development=64,confirmation=64,":"contract=dict(version=1,queries=len(labels),spectra=len(queries),development=0,confirmation=len(labels),group_sha256=group_pin,previous_C08_identity_overlap=0,competition_input_group_overlap=0,",
        "clarkkitchen/casmi26-c08-validation-20261108":"clarkkitchen/casmi26-c09-validation-20261109",
        "CASMI26 C08 external validation 20261108":"CASMI26 C09 disjoint validation 20261109",
        "# Private C08 external validation":"# Private C09 acquisition-disjoint confirmation",
        "128 fixed novel identities;64 development/64 reserved. No candidate truth injection.":"All eligible disjoint identities, confirmation only; exclude previous C08 keys and all training-connected cached input groups. No candidate truth injection.",
        "DONE C08 EXTERNAL VALIDATION PREPARATION":"DONE C09 DISJOINT VALIDATION PREPARATION"
    }
    for before,after in replacements.items():
        assert before in original,before
        original=original.replace(before,after)
    return original


if __name__=='__main__':
    code=source();compile(code,'c09_validation_export','exec')
    exec(compile(code,'c09_validation_export','exec'),dict(__name__='__main__',__file__=str(ROOT/'scripts/prepare_compatible_validation.py')))
