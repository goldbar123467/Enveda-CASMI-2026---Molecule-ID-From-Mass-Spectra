"""Guarded cloud least-squares bookkeeping for published experiment metrics."""
import argparse
import json
import platform
from pathlib import Path
import workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization",required=True)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    proof = workload.require_server_cloud(args.authorization,"cpu")
    root = Path(proof["project_root"]).resolve()
    source,output = Path(args.input).resolve(),Path(args.output).resolve()
    if not source.is_relative_to(root) or not output.is_relative_to(root/"outputs"):
        raise ValueError("Calibration files must stay inside the authorized project")
    rows = json.loads(source.read_text())["rows"]
    import numpy as np
    def fit(group):
        if len(group)<2:
            return {"points":len(group),"fit":None}
        x = np.asarray([[r["class1"],r["class2"]] for r in group],np.float64)
        y = np.asarray([r["public_lb"] for r in group],np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Calibration rows must contain finite published metrics")
        coefficients,_,rank,singular = np.linalg.lstsq(x,y,rcond=None)
        predictions = x@coefficients
        return {"points":len(group),"a":float(coefficients[0]),"b":float(coefficients[1]),"rank":int(rank),
                "residuals":[{"version":r["version"],"predicted_lb":float(p),"actual_lb":float(a),
                              "actual_minus_predicted":float(a-p),"over_0_015":bool(abs(a-p)>0.015)}
                             for r,p,a in zip(group,predictions,y)]}
    result = {"model":"LB = a*C1 + b*C2 (no intercept)","rows":rows,"all":fit(rows),
              "linear":fit([r for r in rows if r["ranking"]=="linear"]),
              "ranker":fit([r for r in rows if r["ranking"]=="learned"]),
              "runtime":"vast-ssh","python":platform.python_version(),"numpy":np.__version__}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({name:{k:v for k,v in result[name].items() if k!="residuals"}
                      for name in ("all","linear","ranker")}),flush=True)


if __name__=="__main__":
    main()
