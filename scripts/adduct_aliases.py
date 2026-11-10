"""Equivalent formate spelling for the credited PV engine's own vocabulary."""


def normalize_pv_adduct(value):
    text=str(value).strip()
    return {'[M+FA-H]-':'[M+CH2O2-H]-'}.get(text,text)
